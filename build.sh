#!/usr/bin/env bash
# Builds AMD FSR 4 (FidelityFX SDK 2.0.0, FSR 4.0.2, MIT) for Vulkan: the INT8 model's passes as SPIR-V for VeyAmux.
#   DXC=/path/to/dxc ./build.sh [version]   ->  dist/veyamux-fsr4-<version>.tar.gz (+ .sha256)
set -euo pipefail
cd "$(dirname "$0")"
DXC="${DXC:-dxc}"
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
    "$DXC" "${COMMON[@]}" "$@" -E "$entry" "$file" -Fo "$OUT/$out.spv" &
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
for j in "${jobs[@]}"; do wait "$j"; done
python3 tools/extract_weights.py amd/fsr4/provider/ffx_provider_fsr4_dx12.cpp "$OUT"
cp amd/LICENSE.txt "$OUT/LICENSE-AMD.txt"
cp NOTICE.md "$OUT/NOTICE.md"
# manifest: what VeyAmux loads (and the sizes AMD's provider uses)
python3 - "$OUT" "$VER" <<'PY'
import hashlib, json, os, sys
out, ver = sys.argv[1], sys.argv[2]
files = {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in sorted(os.listdir(out)) if f != "manifest.json"}
json.dump({"name": "veyamux-fsr4", "version": ver, "fsr": "4.0.2", "model": "fsr4_model_v07_i8",
           "presets": ["native", "quality", "balanced", "performance", "ultraperf", "drs"], "resolutions": [1080, 2160],
           "scratch": {"1080": 20880256, "2160": 83232256}, "files": files},
          open(os.path.join(out, "manifest.json"), "w"), indent=1)
PY
if command -v spirv-val >/dev/null; then
    for f in "$OUT"/*.spv; do spirv-val --target-env vulkan1.3 "$f" || { echo "invalid SPIR-V: $f"; exit 1; }; done
fi
# reproducible: the same sources give the same archive (VeyAmux pins its SHA-256)
tar -C build --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - fsr4 | gzip -n -9 > "dist/veyamux-fsr4-$VER.tar.gz"
( cd dist && sha256sum "veyamux-fsr4-$VER.tar.gz" > "veyamux-fsr4-$VER.tar.gz.sha256" )
echo "dist/veyamux-fsr4-$VER.tar.gz: $(ls "$OUT" | wc -l) files"
