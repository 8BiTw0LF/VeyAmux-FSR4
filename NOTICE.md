# VeyAmux FSR 4 add-on

AMD FidelityFX Super Resolution 4 (FSR 4.0.2, ML upscaling), built for Vulkan so VeyAmux can use it on Linux.

- **Source:** AMD FidelityFX SDK 2.0.0, published by AMD under the MIT license (Copyright © 2025 Advanced Micro
  Devices, Inc.; the license is in `LICENSE-AMD.txt`). Taken from the reference mirror
  https://github.com/Rolaand-Jayz/FSR-4.0.2-reference (commit c5e34b4b7128ffeb152f80cbdb6d715d577ec289).
- **What is used:** both of AMD's models with their weights (`initializers.bin`) and pre- and post-pass shaders:
  the INT8 model (`fsr4_model_v07_i8`, for any GPU with packed 8-bit dot products; its pre-pass weights come from
  AMD's provider source) and the FP8 model (`fsr4_model_v07_fp8_no_scale`, AMD's path for the matrix units of RDNA 4,
  with the border-reset passes AMD's provider runs between its network passes).
- **Changes for Vulkan** (done by `build.sh`, AMD's files are unchanged in `amd/`): the pre-pass weights constant
  buffer gets an explicit register; `QuadReadAcrossX/Y` become subgroup shuffles (no compute-derivative extension
  needed); `_Static_assert` (not in DXC's SPIR-V backend) is dropped; texture element types are 32-bit floats
  (Vulkan's rule for sampled images); storage images written as RGB get four components.
- **The FP8 model on Vulkan:** AMD's FP8 code uses the WaveMatrix (WMMA) API of AMD's D3D12 driver extension.
  `vulkan/AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl` is VeyAmux's own implementation of that API (MIT) on
  standard Vulkan: VK_KHR_cooperative_matrix, VK_EXT_shader_float8 (E4M3) and VK_EXT_cooperative_matrix_maintenance1.
  `tools/spv_float8.py` finishes the SPIR-V (DXC cannot write FP8 types yet). AMD's driver extension header is not
  used or included.
- **Not affiliated with AMD.** FSR and FidelityFX are AMD's. This is not an official AMD release.
