# VeyAmux FSR 4 add-on

AMD FidelityFX Super Resolution 4 (FSR 4.0.2, ML upscaling), built for Vulkan so VeyAmux can use it on Linux.

- **Source:** AMD FidelityFX SDK 2.0.0, published by AMD under the MIT license (Copyright © 2025 Advanced Micro
  Devices, Inc.; the license is in `LICENSE-AMD.txt`). Taken from the reference mirror
  https://github.com/Rolaand-Jayz/FSR-4.0.2-reference (commit c5e34b4b7128ffeb152f80cbdb6d715d577ec289).
- **What is used:** the INT8 model (`fsr4_model_v07_i8`, the version that runs on any GPU with packed 8-bit dot
  products), its weights (`initializers.bin`), the pre- and post-pass shaders and the pre-pass weights from AMD's
  provider source.
- **Changes for Vulkan** (done by `build.sh`, AMD's files are unchanged in `amd/`): the pre-pass weights constant
  buffer gets an explicit register; `QuadReadAcrossX/Y` become subgroup shuffles (no compute-derivative extension
  needed); `_Static_assert` (not in DXC's SPIR-V backend) is dropped; texture element types are 32-bit floats (Vulkan's rule for sampled images).
- **Not affiliated with AMD.** FSR and FidelityFX are AMD's. This is not an official AMD release.
