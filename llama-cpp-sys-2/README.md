# llama-cpp-sys

Raw bindings to llama.cpp, with a cargo feature for each ggml backend.

See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a safe API.

| Feature | Backend | Silicon |
|---|---|---|
| `cuda`, `cuda-no-vmm` | CUDA | NVIDIA |
| `rocm` | HIP | AMD Radeon, Instinct, Ryzen integrated GPUs |
| `metal` | Metal | Apple |
| `vulkan` | Vulkan | any GPU with a Vulkan driver |
| `sycl` | oneAPI SYCL | Intel GPUs |
| `openvino` | OpenVINO | Intel CPU, GPU, NPU |
| `opencl` | OpenCL | Qualcomm Adreno |
| `hexagon` | Hexagon | Qualcomm NPU |
| `cann` | CANN | Huawei Ascend |
| `musa` | MUSA | Moore Threads |
| `webgpu` | WebGPU (Dawn) | any GPU |
| `zdnn` | zDNN | IBM Z |
| `et` | ET | Esperanto ET-SoC |
| `zendnn` | ZenDNN | AMD Zen CPUs |
| `kleidiai` | KleidiAI in the CPU backend | AArch64 CPUs |
| `blas`, `mkl` | BLAS | CPUs |
| `virtgpu` | VirtGPU remoting | a guest of a host with a ggml backend |
| `rpc` | RPC | another machine |

`cuda`, `rocm`, `mkl` and `zendnn` are linked by the build script. The other
vendor backends link a runtime from the vendor's SDK: a static build leaves
that link to the final crate, and a `dynamic-link` build carries it in the
shared libraries. `GGML_*` environment variables are forwarded to CMake.

AVX, AVX2, AVX-512, AVX-VNNI and AMX are enabled from the Rust target
features of the build, or all at once with `-C target-cpu=native`.
