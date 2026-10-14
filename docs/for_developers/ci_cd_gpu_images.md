# GPU container images in CI/CD

Three container images are defined in `.cicd/build-linux/`, with different jobs:

| File | Role | Used by | Size |
|---|---|---|---|
| `runtime-gpu-image.def` | **Run** open-DARTS models on GPUs | users, HPC, `apptainer pull` (see [GPU runtime container](../containers/apptainer.md)) | 1.5 GB (SIF) |
| `builder-gpu-image.def` | **Compile** open-DARTS with CUDA | `apptainer-builder-gpu-image`, `build-linux-tahiti-image-gpu` | 6.3 GB unpacked |
| `Dockerfile` | **Compile** manylinux CPU wheels (GCC 13 on glibc 2.27) | `docker-builder-image` | — |

`Toolchain.cmake` belongs to the `Dockerfile` build.

## The two Apptainer images on the runner

The builder image is the CI compile environment and stays on the tahiti runner; the runtime image ships open-DARTS
installed and is the only image published to the container registry. Both are kept in
`/data/open-darts-apptainer-images` on the runner, with different file-name prefixes: `open-darts-<key>.sif` for the
builder image and `darts-gpu-<key>.sif` for the runtime image. Retention is one rule in `.apptainer-image-store`, which
both jobs extend: keep the newest `MAX_IMAGE_NUM` (5) images of the job's own `IMAGE_PREFIX`, and never delete the
`<prefix>-latest.sif` symlink or what it points at. Because the rule keys off the prefix, neither job can delete the
other's images.

The builder image is keyed on a content hash of `builder-gpu-image.def`, the job definitions and the build arguments, so
an unchanged recipe is never rebuilt; `UPDATE_APPTAINER_IMAGE=1` forces a rebuild. The runtime image also depends on the
source it wraps, so its key is the mode plus `CI_COMMIT_SHA` plus the recipe hash: re-running a pipeline on the same
commit reuses the image, a new commit rebuilds it. `UPDATE_RUNTIME_IMAGE=1` forces a rebuild.

## CI jobs

| Job | Stage | What it does |
|---|---|---|
| `apptainer-builder-gpu-image` | `build_images` | builds the builder SIF, cached by its content hash |
| `build-linux-tahiti-image-gpu` | `build` | compiles open-DARTS **inside the builder SIF** into a wheel |
| `apptainer-runtime-gpu-image` | `build` | assembles the runtime SIF around that wheel |
| `test-apptainer-runtime-gpu` | `test` | `apptainer test --nv`, then the whole suite **inside the runtime SIF** |
| `trigger-testbench` | `test` | with `RUN_TESTBENCH=1`, after `test-apptainer-runtime-gpu`: the model testbench inside the runtime SIF (see *Model testbench* below) |
| `deploy-apptainer-runtime-gpu-image` | `deploy` | publishes `oras://$CI_REGISTRY_IMAGE/darts-gpu` from every `development` pipeline and every release tag |

The builder image compiles; the runtime image runs. The test job deliberately uses the runtime image, so what CI
exercises on every merge request is the artifact users are handed (its venv, its trimmed CUDA set, its runscript and
its GPU self-test) rather than the build environment. There is no conda and no `pip install` anywhere in the test path.

`apptainer-runtime-gpu-image` sits in the `build` stage, not in `build_images` with the other image jobs, because it
needs the wheel of `build-linux-tahiti-image-gpu`, and `needs:` may only name jobs of the same or an earlier stage. The
image installs that wheel, and `ldd` then checks that the CUDA runtime libraries the image installs (a fixed set of
packages from NVIDIA's repository, not copied from the builder image) cover the wheel's shared objects.

`apptainer-runtime-gpu-image` has two modes:

| Mode | When | What it does |
|---|---|---|
| `validation` (default) | merge requests, protected branches other than `development` | reuses the wheel from `build-linux-tahiti-image-gpu`. That wheel is a `native` build for the runner's own GPU, so the image is labelled `CUDA_ARCH: ci-validation-native` and takes ~8 minutes instead of a multi-architecture compile. |
| `portable` | `development` pipelines, release tags, `RUN_RUNTIME_IMAGE=1`, `RUN_RUNTIME_IMAGE_DEPLOY=1` | compiles the multi-architecture wheel first. This is the artifact that gets published. |

`deploy-apptainer-runtime-gpu-image` reads the `CUDA_ARCH` label off the image and refuses to publish anything that was
not built in portable mode, so a single-GPU validation image cannot reach the registry by accident.

### Choosing a GPU

Every GPU job chooses its GPU with `helper_scripts/gpu_ci_preflight.py`: it samples `nvidia-smi` for 10 s and takes
the least-loaded GPU with at least 4 GiB free, the lowest index on a tie (`GPU_DEVICE` overrides the choice). The
testbench GPU job, which measures performance, also claims its GPU (`DARTS_GPU_CLAIM=1`): holding a lock on the claim
directory, `/tmp/open-darts-gpu-claims` (`DARTS_GPU_CLAIM_DIR`), it takes the least-loaded unclaimed GPU, the highest
index on a tie, and leaves an empty file `<gpu-uuid>.<pgid>` there. Every other preflight on the runner then takes that
GPU only if no other is left. A job that chose the same GPU just before, too early to see the claim, shows up as a new
CUDA process while the claimant watches its GPU for 15 s; the claim then moves to a GPU without processes or claims,
never back, or stays if there is none. A claim lasts as long as the job's process group, 12 h at most, and the claims
of ended jobs are ignored and pruned, so no job has to clean up. A claim that cannot be written, or is lost because
`/tmp` was emptied, only costs the reservation: the job still runs, and the testbench does not compare the timings of
runs that shared their GPU with their references. Two jobs of the same kind never compete: `resource_group` runs one
`test-apptainer-runtime-gpu` and one testbench GPU job at a time on tahiti.

## Pipeline variables

Set for one pipeline in the *Run pipeline* form, through the API or in a pipeline schedule:

| Variable | Default | Effect |
|---|---|---|
| `RUN_RUNTIME_IMAGE` | unset | `1`: build the runtime image in portable mode in any pipeline; nothing is published |
| `RUN_RUNTIME_IMAGE_DEPLOY` | unset | `1`: build the runtime image in portable mode and, on a protected ref, publish it as `<sha12>` |
| `UPDATE_APPTAINER_IMAGE` | unset | `1`: rebuild the builder image even if one with the same key exists |
| `UPDATE_RUNTIME_IMAGE` | unset | `1`: rebuild the runtime image even if one with the same key exists |
| `RUN_TESTBENCH` | `0` | `1`: `trigger-testbench` runs the model testbench in this pipeline's runtime image (set it for a scheduled, manual or API pipeline) |
| `TESTBENCH_MODELS` | empty: all models | testbench models to run, space-separated |
| `TESTBENCH_TIER` | `nightly weekly` | testbench tiers to run: all but the smoke tests (`mr`), the hours-long `full` tier and the unvalidated `manual` variants |
| `TESTBENCH_VTK` | `1` | `1`: a VTK snapshot of every `DartsModel.run()` and an animation per variant, kept as testbench artefacts |

The `RUN_TESTBENCH` and `TESTBENCH_*` defaults are set in `.gitlab-ci.yml`. On its own the testbench runs the full
models of `spe10` (tiers `nightly weekly` without the smoke tests, i.e. `spe10` `full`) without VTK output.
The other variables of the testbench's `gpu` job, such as `TESTBENCH_VARIANTS` or `TESTBENCH_EXCLUDE_VARIANTS`, can be
set for a pipeline as well and are forwarded to it.

Job variables, which a pipeline can override as well:

| Variable | Default | Effect |
|---|---|---|
| `BUILDER_CUDA_VERSION` | `12.8.1` | CUDA toolkit of the builder image |
| `ENABLE_AGENT_TOOLS` | `0` | `1`: adds the agent tooling to the builder image |
| `RUNTIME_CUDA_VERSION` | `12.8.1` | CUDA runtime of the runtime image |
| `RUNTIME_CUDA_ARCH` | `70-real;80-real;90-real;100-real;120-real;120-virtual` | GPU targets of a portable build |
| `RUNTIME_CPU_ARCH` | `x86-64-v3` | `-march` baseline of a portable build |
| `MAX_IMAGE_NUM` | `5` | images of each prefix kept on the runner |

The runtime image's Python is the CI's default Python, the project variable `DEFAULT_PY_VER`, and the wheel's ABI tag is
checked against it.

## Model testbench

`trigger-testbench` (`.cicd/jobs/testbench.yml`) starts a pipeline of
[open-darts/testbench](https://gitlab.com/open-darts/testbench) whose GPU job runs on tahiti inside this pipeline's
runtime image (it checks that the image's `VCS_REF` is this commit) and runs the variants selected by the `TESTBENCH_*`
variables. With `strategy: depend` the testbench result decides the result of this pipeline. The job exists exactly in
the pipelines that have `apptainer-runtime-gpu-image`, and starts only once `test-apptainer-runtime-gpu` of the same
pipeline (same stage, through `needs:`) has passed, where that job exists: the testbench then shares tahiti with
neither the GPU tests' CPU load nor a build whose GPU tests failed.

The testbench GPU job claims its GPU (*Choosing a GPU* above): the other CI jobs on the runner then choose another GPU,
so its performance is measured on a GPU of its own.

## Building the runtime image

```bash
.cicd/build-linux/build_runtime_image.sh
```

Two phases:

1. compile a **portable** wheel inside the builder image: multi-architecture cubins plus a PTX fallback instead of
   CMake's `native`, and an explicit `-march` baseline instead of `-march=native`;
2. assemble the runtime image around that wheel.

Skip phase 1 with `--wheel` when a suitable wheel already exists (this is what CI does with the
`build-linux-tahiti-image-gpu` artifact):

```bash
.cicd/build-linux/build_runtime_image.sh --wheel dist/open_darts-*.whl
```

Useful options (`--help` lists them all):

| Option | Effect |
|---|---|
| `--cuda-version 13.0.2` | smaller image, needs r580+, no `sm_70` |
| `--cuda-arch 80` | single-architecture image, ~600 MB smaller, A100 only |
| `--cpu-arch native` | fastest host code, but the image stops being portable |
| `--no-cuda-compat` | drops NVIDIA's forward-compat `libcuda` (−307 MB) |
| `--extras test,viz` | adds vtk/pyvista (+~150 MB) |
| `--no-cudss` | leaves out the cuDSS direct solver (−103 MB of SIF) |

```{note}
**Build on local disk.** `apptainer build` unpacks the base image with extended attributes, which NFS rejects
(`lclearxattrs: permission denied`). The script defaults `APPTAINER_TMPDIR` to `${TMPDIR:-/var/tmp}`; override it with
another local path if `/var/tmp` is small. `mksquashfs` with zstd support is optional: it gives a smaller SIF, and
without it the image is compressed with Apptainer's default gzip.
```

### Portability, concretely

The stock GPU build is not relocatable, for two independent reasons. Both are parameters rather than hardcoded:

* `CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which emits SASS for the build host's GPU **and no PTX**:
  `cuobjdump --list-ptx` on a stock GPU build reports "No PTX file found". An image built on an A100 will therefore not
  start on an H100. `CUDA_ARCH` overrides it; the default list of `build_runtime_image.sh` yields `sm_70 sm_80 sm_90
  sm_100 sm_120` cubins plus `compute_120` PTX. With the four architectures `sm_70 sm_80 sm_90 sm_120` plus PTX,
  `engines.so` grew from 124 MB to 282 MB (AMGX is statically linked into it, and grows the same way), and everything
  open-DARTS installs to 407 MB.
* GPU builds appended `-march=native`, baking the build host's CPU ISA into every object, so a container built on a
  newer CPU dies with `SIGILL` on an older one. The `CPU_ARCH` CMake variable overrides it: it still defaults to
  `native`, so nothing changes for local builds; `CPU_ARCH=off` omits `-march`.

Both are plumbed through `helper_scripts/build_darts_cmake.sh` as environment variables, so they work for ordinary
builds too:

```bash
CUDA_ARCH="80-real;90-real" CPU_ARCH=x86-64-v3 ./helper_scripts/build_install_darts_gpu.sh -c -p
```

### Where the size goes

Measured inside an image with `sm_70 sm_80 sm_90 sm_120` cubins plus PTX (CUDA 12.8, `--extras test`), which
compresses to a **1.5 GB** SIF:

| Component | Uncompressed |
|---|---|
| `libcublasLt.so.12` | 717 MB |
| `libcusparse.so.12` | 369 MB |
| `libcusolver.so.11` | 230 MB |
| `libcublas.so.12` | 111 MB |
| `libnvJitLink.so.12` | 90 MB |
| `libcudart.so.12`, `libnvblas.so.12` | 2 MB |
| **CUDA total** | **1518 MB** |
| NVIDIA forward-compat `libcuda` | 193 MB (drop with `--no-cuda-compat`) |
| venv, of which open-DARTS binaries are 407 MB | 1436 MB |
| Rocky 8 + Python 3.12 | ~250 MB |
| cuDSS `libcudss.so.0` in the wheel, added since (not in the 1.5 GB) | 149 MB; the SIF grows by 103 MB (drop with `--no-cudss`) |

The same set on CUDA 13.0 is 1000 MB, the difference being almost entirely cuBLASLt, cuSPARSE and cuSOLVER.

cuDSS comes from NVIDIA's `nvidia-cudss-cu12` pip wheel, pinned in `build_runtime_image.sh` and in
`build-linux-tahiti-image-gpu`: every 0.x minor release so far has changed its API under the same soname,
`libcudss.so.0`, so a new version comes in as a bump that the MR's GPU jobs test. Its CUDA 12 build carries cubins for `sm_50` to `sm_120` plus `sm_120` PTX, so it covers
every GPU of the image, and it needs only `libcublas.so.12`. NVIDIA's licence allows redistributing it only as part of
an application and with their notice: the wheel bundles `LICENSE.cuDSS.txt` next to the library, and the image build
fails without it.

`libcublasLt` is the single biggest item and cannot be removed: nothing in open-DARTS calls it directly, but
`libcublas` has it as a hard `DT_NEEDED`. NVIDIA's `nvprune` accepts only relocatable objects and archives, so the
per-architecture cubins inside the stock shared libraries cannot be trimmed.

NVIDIA splits headers and static archives into the `-devel` subpackages, so the runtime RPMs bring almost no dead weight
of their own; the `*.a` sweep in `%post` is a guard in case that changes. Measured on CUDA 13.0, the trim takes the
toolkit from 1103 MB to 1000 MB, essentially all of it `libcusolverMg`.

The set of CUDA libraries is not guesswork: it is the `DT_NEEDED` closure of the built `engines`, `linear_solvers` and
AMGX objects, and `runtime-gpu-image.def` re-verifies that closure with `ldd` during `%post`. A future commit that starts
using, say, cuFFT fails the image build instead of failing at a user's first run.

## Building the builder image

```bash
APPTAINER_TMPDIR=/var/tmp/ap apptainer build --fakeroot \
    open-darts-builder.sif .cicd/build-linux/builder-gpu-image.def
```

`--build-arg CUDA_VERSION=13.0.2` matches it to a runner with a different toolkit; `--build-arg ENABLE_AGENT_TOOLS=1`
adds the agent tooling.

Its `%test` section compiles and links a small CUDA program, which guards the static-archive trim in `%post`: the CUDA
devel image ships ~2.3 GB of static archives and open-DARTS links exactly two of them (`libcudart_static.a` and
`libculibos.a`), so the rest is deleted.

## Troubleshooting image builds

**`fatal error: stddef.h: No such file or directory` from a compile inside the builder image**: the host environment
leaked in. Apptainer passes the caller's variables into the container, and with a spack/lmod gcc module loaded outside,
`GCC_ROOT` points at a host prefix that the container's own gcc then uses to locate its internal headers. The headers
are present; gcc is looking in the wrong place. `builder-gpu-image.def`'s `%environment` overrides `GCC_ROOT`, `CC`,
`CXX` and `CMAKE_PREFIX_PATH` to the image's own toolchain, and its `%test` asserts the override held.
`apptainer exec --cleanenv` is the general remedy, and `DARTS_KEEP_HOST_TOOLCHAIN_ENV=1` opts out of the override.

**`lclearxattrs: permission denied` while building**: `APPTAINER_TMPDIR` is on NFS. Point it at local disk.

**Editing an `%environment` section**: Apptainer collects the variables the section *exports* and lays them over the
inherited environment. An `unset` there is silently ignored, so a host variable you want gone has to be **overridden**
with a sane value instead. Only `%runscript` can really unset one, and only for `apptainer run`.

**Editing a `%post` section**: Apptainer's default `mount tmp = yes` bind-mounts the host's `/tmp` *and* `/var/tmp` into
`%post`, so `rm -rf /tmp/*` there deletes the build machine's temp files rather than the image's; on a shared runner
that is everyone's, and it also wipes `APPTAINER_TMPDIR` out from under the build in progress. `/root` and `$HOME` are
container-local, so those are safe. Verify with `cat /proc/mounts` inside a throwaway `%post` if in doubt.
