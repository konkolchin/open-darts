# Container images

- `runtime-gpu-image.def`, built by `build_runtime_image.sh`: the GPU runtime image that users run, see
  [docs/containers/apptainer.md](../../docs/containers/apptainer.md).
- `builder-gpu-image.def`: the CI image that compiles open-DARTS with CUDA.
- `Dockerfile`, with `Toolchain.cmake`: the CI image that compiles the manylinux CPU wheels.

How the images are built, their CI jobs and the pipeline variables that control them:
[docs/for_developers/ci_cd_gpu_images.md](../../docs/for_developers/ci_cd_gpu_images.md).
