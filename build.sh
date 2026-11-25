#!/usr/bin/env bash
# Builds AMD FSR 4 (FidelityFX SDK 2.0.0, FSR 4.0.2, MIT) for Vulkan: both of AMD's networks as SPIR-V for VeyAmux,
# the INT8 one (any GPU with packed 8-bit dot products) and the FP8 one (matrix units, RDNA 4).
#   DXC=/path/to/dxc SPIRV_VAL=/path/to/spirv-val ./build.sh [version]   ->  dist/veyamux-fsr4-<version>.tar.gz (+ .sha256)
# SPIRV_VAL must know SPV_EXT_float8 and SPV_EXT_cooperative_matrix_maintenance1 (SPIRV-Tools from 2025 on).
# EMULATE=1 builds the FP8 shaders on an emulation of the matrix units instead (tests/emulate; tests only, not
# published): build/fsr4 then runs anywhere, e.g. on lavapipe, with VEYAMUX_FSR4_FP8=emulate.
set -euo pipefail
cd "$(dirname "$0")"
DXC="${DXC:-dxc}"
SPIRV_VAL="${SPIRV_VAL:-spirv-val}"
EMULATE="${EMULATE:-0}"
VER="${1:-dev}"
OUT="$PWD/build/fsr4"
rm -rf build && mkdir -p "$OUT" dist build/src

# AMD's sources, with the few changes Vulkan needs (patches/README.md says why)
cp -r amd/fsr4/. build/src/
sed -i 's/^cbuffer cbPass_Weights$/cbuffer cbPass_Weights : register(b1)/' build/src/include/pre_common.hlsli
grep -q 'cbPass_Weights : register(b1)' build/src/include/pre_common.hlsli
# Vulkan: a storage image write must give every component of the format, so the RGB targets become float4
R=build/src/include/ffx_fsr4_upscale_resources.h
sed -i -E 's/^RWTexture2D<(float3|DATA_TYPE3)>( +)(rw_[a-z_]+)/RWTexture2D<float4>\2\3/' "$R"
sed -i 's/rw_mlsr_output_color\[iPxPos\] = color;/rw_mlsr_output_color[iPxPos] = float4(color, 1);/' build/src/include/mlsr_optimized_includes.hlsli
sed -i 's/rw_history_color\[tid\] = model_color;/rw_history_color[tid] = float4(model_color, 1);/; s/rw_mlsr_output_color\[tid\] = final_color;/rw_mlsr_output_color[tid] = float4(final_color, 1);/' build/src/include/post_common.hlsli
sed -i 's/rw_reprojected_color\[dtid.xy\] = DATA_TYPE3(history);/rw_reprojected_color[dtid.xy] = float4(history, 1);/' build/src/include/pre_common.hlsli
for want in "float4(color, 1)" "float4(model_color, 1)" "float4(final_color, 1)" "float4(history, 1)"; do
    grep -rq "$want" build/src/include || { echo "patch failed: $want"; exit 1; }
done
grep -q 'RWTexture2D<float3>' "$R" && { echo "patch failed: float3 storage images left"; exit 1; } || true
# FP8 (WMMA) sources: the same image writes in their own pre/post passes
sed -i 's/rw_reprojected_color\[dtid.xy\] = DATA_TYPE3(history);/rw_reprojected_color[dtid.xy] = float4(history, 1);/' build/src/shaders/pre_wmma.hlsl
sed -i 's/rw_history_color\[tid\] = model_color;/rw_history_color[tid] = float4(model_color, 1);/; s/rw_mlsr_output_color\[tid\] = final_color;/rw_mlsr_output_color[tid] = float4(final_color, 1);/' build/src/shaders/post_wmma.hlsl
grep -q 'float4(history, 1)' build/src/shaders/pre_wmma.hlsl && grep -q 'float4(final_color, 1)' build/src/shaders/post_wmma.hlsl || { echo "patch failed: wmma image writes"; exit 1; }
# a zero-sized shared array (ML2C_GROUPSHARED_SIZE 0) is not valid SPIR-V: at least one element
sed -i 's|^groupshared uint ml2c_ldsScratch\[ML2C_GROUPSHARED_SIZE/4\];|groupshared uint ml2c_ldsScratch[(ML2C_GROUPSHARED_SIZE) >= 4 ? (ML2C_GROUPSHARED_SIZE)/4 : 1];|' build/src/ml2code_runtime/storage.hlsli
grep -q 'ML2C_GROUPSHARED_SIZE)/4 : 1' build/src/ml2code_runtime/storage.hlsli || { echo "patch failed: ml2c_ldsScratch"; exit 1; }
# the padding passes: DXC cannot brace-initialize this derived class for SPIR-V; the same with an assignment
for f in build/src/shaders/fsr4_model_v07_fp8_no_scale_passes_*.hlsl; do
    sed -i -E 's/^(    StoreDefaultConstBatchOperation *<.*> *batchOp_([0-9]+)) = \{ DefaultDword \};/\1; batchOp_\2.DWord = DefaultDword;/' "$f"
    grep -q '= { DefaultDword }' "$f" && { echo "patch failed: StoreDefaultConstBatchOperation in $f"; exit 1; } || true
done
# AMD's WaveMatrix API on Vulkan cooperative matrices (vulkan/), or emulated (tests/emulate)
WMMA_INC="$PWD/vulkan"
if [ "$EMULATE" = 1 ]; then
    WMMA_INC="$PWD/tests/emulate"
    # the emulation spreads a matrix over the group's 32 invocations, whatever the device's subgroup size
    for f in build/src/shaders/pre_wmma.hlsl build/src/shaders/post_wmma.hlsl build/src/shaders/fsr4_model_v07_fp8_no_scale_passes_*.hlsl \
             build/src/ml2code_runtime/operators/float8_NHWC/*.hlsli build/src/ml2code_runtime/operators/float8_NHWC/Fused/*.hlsli; do
        sed -i 's/WaveGetLaneIndex()/emu::lane()/g' "$f"
    done
fi

COMMON=(-spirv -fspv-target-env=vulkan1.3 -fspv-entrypoint-name=main -T cs_6_6 -HV 2021 -enable-16bit-types -O3 -no-warnings
        -fvk-t-shift 0 0 -fvk-u-shift 32 0 -fvk-b-shift 64 0 -fvk-s-shift 80 0
        -I build/src -I build/src/include
        "-DQuadReadAcrossX(v)=WaveReadLaneAt(v,WaveGetLaneIndex()^1)"
        "-DQuadReadAcrossY(v)=WaveReadLaneAt(v,WaveGetLaneIndex()^2)"
        "-D_Static_assert(c,m)="
        -DFFX_GPU=1 -DFFX_HLSL=1 -DFFX_MLSR_COLORSPACE=0 -DFFX_MLSR_LOW_RES_MV=1 -DFFX_MLSR_DEPTH_INVERTED=0
        -DFFX_MLSR_AUTOEXPOSURE_ENABLED=0 -DFFX_MLSR_JITTERED_MOTION_VECTORS=0 -DFFX_DEBUG_VISUALIZE=0
        -DDATA_TYPE=float -DDATA_TYPE2=float2 -DDATA_TYPE3=float3 -DDATA_TYPE_VECTOR=float4 -DNETWORK_DATA_TYPE=float16_t)
jobs=()
compile() {   # out entry file defines...
    local out="$1" entry="$2" file="$3"; shift 3
    "$DXC" "${COMMON[@]}" "$@" -E "$entry" "$file" -Fo "$OUT/$out.spv" || { echo "DXC failed: $out"; return 1; } &
    jobs+=($!)
    if (( ${#jobs[@]} >= $(nproc) )); then wait "${jobs[0]}"; jobs=("${jobs[@]:1}"); fi
}
for p in native quality balanced performance ultraperf drs; do
    for r in 1080:0 2160:1; do
        res="${r%%:*}"; idx="${r##*:}"
        d="build/src/shaders/i8_$p"
        compile "i8_${p}_${res}_pre" main "$d/pre.hlsl" -DFFX_MLSR_RESOLUTION=$idx
        compile "i8_${p}_${res}_post" main "$d/post.hlsl" -DFFX_MLSR_RESOLUTION=$idx
        for i in $(seq 1 12); do
            compile "i8_${p}_${res}_pass$i" "fsr4_model_v07_i8_pass$i" "$d/passes_$res.hlsl" -DFFX_MLSR_RESOLUTION=$idx -DMLSR_PASS_$i=1
        done
    done
    cp "amd/fsr4/shaders/i8_$p/initializers.bin" "$OUT/i8_${p}_initializers.bin"
done
# FP8: one set of shaders per resolution class (the presets differ only in their weights). Matrix loads and stores
# go through StructuredBuffer aliases of the weights (WMMA_SRV) and scratch (WMMA_UAV) buffers; their bindings are
# those of InitializerBuffer and ScratchBuffer in each file (t1/u1 in the network passes, t18/u11 in pre/post).
FP8=(-fspv-use-vulkan-memory-model -Vd -I "$WMMA_INC" -DWMMA_ENABLED=1 -DFP8_ENABLED=1 -DDOT4_ENABLED=1)
for r in 1080:0 2160:1; do
    res="${r%%:*}"; idx="${r##*:}"
    f="build/src/shaders/fsr4_model_v07_fp8_no_scale_passes_$res.hlsl"
    compile "fp8_${res}_pre.raw" fsr4_model_v07_fp8_no_scale_prepass build/src/shaders/pre_wmma.hlsl "${FP8[@]}" -DWMMA_SRV_BINDING=18 -DWMMA_UAV_BINDING=43 -DFFX_MLSR_RESOLUTION=$idx
    compile "fp8_${res}_post.raw" fsr4_model_v07_fp8_no_scale_postpass build/src/shaders/post_wmma.hlsl "${FP8[@]}" -DWMMA_SRV_BINDING=18 -DWMMA_UAV_BINDING=43 -DFFX_MLSR_RESOLUTION=$idx
    for i in $(seq 1 12); do
        compile "fp8_${res}_pass$i.raw" "fsr4_model_v07_fp8_no_scale_pass$i" "$f" "${FP8[@]}" -DWMMA_SRV_BINDING=1 -DWMMA_UAV_BINDING=33 -DMLSR_PASS_$i=1
    done
    for i in $(seq 0 12); do
        [ "$i" = 6 ] && continue   # AMD's provider has no border reset after pass 6
        compile "fp8_${res}_pad$i.raw" "fsr4_model_v07_fp8_no_scale_pass${i}_post" "$f" "${FP8[@]}" -DWMMA_SRV_BINDING=1 -DWMMA_UAV_BINDING=33 -DMLSR_PASS_${i}_POST=1
    done
done
for p in native quality balanced performance ultraperf drs; do
    cp "amd/fsr4/shaders/fp8_$p/initializers.bin" "$OUT/fp8_${p}_initializers.bin"
done
for j in "${jobs[@]}"; do wait "$j"; done
# the FP8 shaders' stand-ins (see tools/spv_float8.py) become E4M3 and the use conversion
for f in "$OUT"/*.raw.spv; do
    python3 tools/spv_float8.py "$f" "${f%.raw.spv}.spv" > /dev/null
    rm "$f"
done
python3 tools/extract_weights.py amd/fsr4/provider/ffx_provider_fsr4_dx12.cpp "$OUT"
cp amd/LICENSE.txt "$OUT/LICENSE-AMD.txt"
cp NOTICE.md "$OUT/NOTICE.md"
# manifest: what VeyAmux loads (and the sizes AMD's provider uses)
python3 - "$OUT" "$VER" <<'PY'
import hashlib, json, os, sys
out, ver = sys.argv[1], sys.argv[2]
files = {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in sorted(os.listdir(out)) if f != "manifest.json"}
json.dump({"name": "veyamux-fsr4", "version": ver, "fsr": "4.0.2", "model": "fsr4_model_v07_i8",
           "fp8": {"model": "fsr4_model_v07_fp8_no_scale", "emulated": os.environ.get("EMULATE") == "1"},
           "presets": ["native", "quality", "balanced", "performance", "ultraperf", "drs"], "resolutions": [1080, 2160],
           "scratch": {"1080": 20880256, "2160": 83232256}, "files": files},
          open(os.path.join(out, "manifest.json"), "w"), indent=1)
PY
for f in "$OUT"/*.spv; do "$SPIRV_VAL" --target-env vulkan1.3 "$f" || { echo "invalid SPIR-V: $f"; exit 1; }; done
# reproducible: the same sources give the same archive (VeyAmux pins its SHA-256)
tar -C build --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - fsr4 | gzip -n -9 > "dist/veyamux-fsr4-$VER.tar.gz"
( cd dist && sha256sum "veyamux-fsr4-$VER.tar.gz" > "veyamux-fsr4-$VER.tar.gz.sha256" )
echo "dist/veyamux-fsr4-$VER.tar.gz: $(ls "$OUT" | wc -l) files"
