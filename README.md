# VeyAmux FSR 4 add-on

AMD FidelityFX Super Resolution 4 (FSR 4.0.2, machine-learning upscaling) built for Vulkan, so that
[VeyAmux](https://github.com/8BiTw0LF/VeyAmux) can use it on Linux. VeyAmux downloads a release of this repository
when you choose **FSR 4** in its menu, checks it against a SHA-256 it knows, and installs it into
`~/.local/share/VeyAmux/fsr4/`. Nothing here is needed or used by anything else.

- **What is in it:** AMD's two FSR 4 models, compiled to SPIR-V with DXC, with their weights, network passes and
  pre/post passes. VeyAmux's own code runs them (`src/Fsr4Upscaler.cpp` in VeyAmux) and picks one:
  - **FP8** (`fsr4_model_v07_fp8_no_scale`): AMD's own path on RDNA 4, 16x16 FP8 matrix multiplies on the matrix
    units through standard Vulkan (VK_KHR_cooperative_matrix, VK_EXT_shader_float8,
    VK_EXT_cooperative_matrix_maintenance1; RADV in Mesa 26.3 or newer).
  - **INT8** (`fsr4_model_v07_i8`): runs on AMD, NVIDIA and Intel GPUs with packed 8-bit dot products.
- **Where it comes from:** AMD FidelityFX SDK 2.0.0, which AMD published under the MIT license. See [NOTICE.md](NOTICE.md)
  for the exact source and the few changes made for Vulkan; AMD's files are unchanged in `amd/`.
- **Not from AMD.** FSR and FidelityFX are AMD's. This is an unofficial build, kept apart from VeyAmux on purpose.

## Building

```sh
DXC=/path/to/dxc SPIRV_VAL=/path/to/spirv-val ./build.sh 2     # -> dist/veyamux-fsr4-2.tar.gz (+ .sha256), reproducible
```

DXC 1.9.2609 (the same as VeyAmux) and a `spirv-val` that knows SPV_EXT_float8 and
SPV_EXT_cooperative_matrix_maintenance1 (SPIRV-Tools from late 2025 on; CI builds a pinned one). A tag `vN` makes
GitHub Actions build and publish `veyamux-fsr4-N.tar.gz`; VeyAmux then needs that archive's SHA-256 in
`src/Fsr4Addon.cpp`.

### Testing the FP8 path without RDNA 4

`EMULATE=1 ./build.sh emu` builds the FP8 shaders on `tests/emulate/`, a plain-HLSL emulation of the matrix units
(slow, test only, never published). VeyAmux runs them on any Vulkan device with `VEYAMUX_FSR4_FP8=emulate` and
`VEYAMUX_FSR4_DIR=build/fsr4`, e.g. its `veyamux-upscale-test ... fsr4` (FP8) against `... fsr4i8` (INT8) on
lavapipe: the two networks should give nearly the same picture.

## License

AMD's code and weights: MIT (`amd/LICENSE.txt`). The build scripts here: MIT as well.
