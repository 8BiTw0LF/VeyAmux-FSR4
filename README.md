# VeyAmux FSR 4 add-on

AMD FidelityFX Super Resolution 4 (FSR 4.0.2, machine-learning upscaling) built for Vulkan, so that
[VeyAmux](https://github.com/8BiTw0LF/VeyAmux) can use it on Linux. VeyAmux downloads a release of this repository
when you choose **FSR 4** in its menu, checks it against a SHA-256 it knows, and installs it into
`~/.local/share/VeyAmux/fsr4/`. Nothing here is needed or used by anything else.

- **What is in it:** AMD's FSR 4 INT8 model (`fsr4_model_v07_i8`, the version that runs on AMD, NVIDIA and Intel GPUs
  with packed 8-bit dot products): its weights, its network passes and the pre/post passes, compiled to SPIR-V with
  DXC. VeyAmux's own code runs them (`src/Fsr4Upscaler.cpp` in VeyAmux).
- **Where it comes from:** AMD FidelityFX SDK 2.0.0, which AMD published under the MIT license. See [NOTICE.md](NOTICE.md)
  for the exact source and the few changes made for Vulkan; AMD's files are unchanged in `amd/`.
- **Not from AMD.** FSR and FidelityFX are AMD's. This is an unofficial build, kept apart from VeyAmux on purpose.

## Building

```sh
DXC=/path/to/dxc ./build.sh 1     # -> dist/veyamux-fsr4-1.tar.gz (+ .sha256), reproducible
```

DXC 1.9.2609 (the same as VeyAmux) and `spirv-val` (SPIRV-Tools). A tag `vN` makes GitHub Actions build and publish
`veyamux-fsr4-N.tar.gz`; VeyAmux then needs that archive's SHA-256 in `src/Fsr4Addon.cpp`.

## License

AMD's code and weights: MIT (`amd/LICENSE.txt`). The build scripts here: MIT as well.
