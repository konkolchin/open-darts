# GPU runtime container (Apptainer)

The open-DARTS GPU runtime image is an [Apptainer](https://apptainer.org) image (a single SIF file) with open-DARTS
installed for NVIDIA GPUs. It runs on any Linux host with an AVX2 CPU and an NVIDIA driver r525 or newer, whatever CUDA
toolkit, distribution or glibc the host has: the image brings its own CUDA runtime and Python, and Apptainer adds the
host's driver. How the image is built and tested is described in
[GPU container images in CI/CD](../for_developers/ci_cd_gpu_images.md).

## Pull and run

```bash
apptainer pull oras://registry.gitlab.com/open-darts/open-darts/darts-gpu:latest
apptainer run --nv darts-gpu_latest.sif model.py
```

`--nv` is not optional: it is how Apptainer injects the host's NVIDIA driver. Without it the image refuses to start
and tells you so.

```bash
# any command inside the environment
apptainer exec --nv darts-gpu.sif python -c "from importlib.metadata import version; print(version('open-darts'))"
apptainer exec --nv darts-gpu.sif darts --help

# check that the image and this host agree, before queueing a long job
apptainer test --nv darts-gpu.sif        # imports darts and runs a real CUDA kernel
apptainer inspect darts-gpu.sif          # CUDA version, GPU targets, CPU baseline, git ref

# data outside $HOME and $PWD has to be bind-mounted
apptainer run --nv -B /scratch/cases:/cases darts-gpu.sif /cases/run.py

# pick a GPU (the APPTAINERENV_ prefix passes a variable through)
APPTAINERENV_CUDA_VISIBLE_DEVICES=1 apptainer run --nv darts-gpu.sif model.py

# threads: open-DARTS honours OMP_NUM_THREADS, and the image does not set it
APPTAINERENV_OMP_NUM_THREADS=16 apptainer run --nv darts-gpu.sif model.py
```

### Tags

| Pipeline | Tags pushed |
|---|---|
| every `development` pipeline (push, web, schedule) | `latest`, `<sha12>` |
| release tag `vX.Y.Z` | `X.Y.Z`, `<sha12>` |
| other protected branch with `RUN_RUNTIME_IMAGE_DEPLOY=1` | `<sha12>` only |

`latest` is the newest `development` build, `X.Y.Z` pins a release, and `<sha12>` names an exact commit. `latest` only
ever moves forward: a pipeline claims it only while its commit is still the tip of `development`, so a slow pipeline
finishing late, or an old one being re-run, cannot roll it back. Merge-request pipelines never publish.

### Registry access

The project is public, but its container registry is currently set to *Only project members*, so an anonymous pull is
refused with `403 access forbidden`. Until a maintainer switches **Settings → General → Visibility → Container
registry** to *Everyone With Access*, log in first with a personal access token that has the `read_registry` scope:

```bash
apptainer registry login -u <gitlab-username> -p <token> oras://registry.gitlab.com
```

That setting covers every image in this project's registry, not just this one.

## Support matrix

The default image is built against **CUDA 12.8**, the only toolkit that covers every GPU the project runs on: CUDA 13
dropped Volta, and CUDA 12.7 and earlier predate Blackwell.

**Nothing about the host has to match the image except the driver and the CPU.** The image carries its own CUDA
runtime and its own Python, and `--nv` injects only the host *driver*, so the host's CUDA toolkit, distro and glibc are
all irrelevant. There is no per-machine rebuild.

The image ships SASS for one compute-capability major each, which by NVIDIA's binary-compatibility rule (*a cubin for
X.y runs on X.z when z ≥ y*) covers the minor revisions above it:

| Shipped | Covers | Examples |
|---|---|---|
| `sm_70` cubin | 7.0, 7.2, 7.5 | V100, Xavier, T4, RTX 20xx |
| `sm_80` cubin | 8.0, 8.6, 8.7, 8.9 | A100, A10, A40, RTX 30xx, Orin, L4, L40S, RTX 40xx |
| `sm_90` cubin | 9.0 | H100, H200, GH200 |
| `sm_100` cubin | 10.x | B200, GB200 |
| `sm_120` cubin | 12.0 | RTX 50xx, RTX PRO Blackwell |
| `compute_120` PTX | newer than 12.0 | JIT on first run, cached in `~/.nv/ComputeCache` |

Not covered: compute capability **below 7.0** (P100, Maxwell). The PTX fallback only works *upward*: it cannot be JIT'd
down to an older architecture, which is why 10.x needs its own cubin rather than relying on it.

| | |
|---|---|
| Minimum driver | **r525** (CUDA 12 minor-version compatibility). Older still works on datacenter GPUs via the bundled forward-compat `libcuda`, which the runscript engages by itself. |
| Host CUDA toolkit | irrelevant: the image brings its own |
| Host distro / glibc | irrelevant: the image brings its own userspace |
| CPU baseline | `x86-64-v3`: AVX2/FMA, i.e. Intel Haswell (2013) / AMD Zen and later |
| Python | inside the image; `apptainer exec darts-gpu.sif python --version` shows which |
| GPU direct solver | cuDSS (NVIDIA's CUDA 12 build, all the GPUs above): `self.linear_solver.spec = CuDSSSolverSpec()`; NVIDIA's licence lies next to it, in `darts/linear_solvers/LICENSE.cuDSS.txt` |

A `CUDA_VERSION=13.0.2` image is about 500 MB smaller (its CUDA libraries are 1000 MB against 1518 MB), but needs an
**r580** driver and drops `sm_70`.

Concretely: this one CUDA 12.8 image runs unchanged on the tahiti runner (CUDA 12.6) and on the oahu/samoa machines
(CUDA 13.3). Preferring CUDA 12 here despite the newer runners is what buys the wider driver floor and Volta.

If the host driver is older than the container's CUDA needs, the runscript falls back to NVIDIA's bundled
forward-compatibility `libcuda` (datacenter GPUs only) and says so. Force it with `DARTS_CUDA_COMPAT=1`, disable with
`DARTS_CUDA_COMPAT=0`.

## Known limitation: Reaktoro

Models using the **Reaktoro** chemistry backend do not work in this image. Reaktoro ships only through conda-forge,
has no PyPI wheel, and the runtime image deliberately has no conda. **IPhreeqc** chemistry does work: `libIPhreeqc.so`
is bundled into the wheel and `phreeqpy` comes from PyPI. For Reaktoro, use a conda environment.

## Troubleshooting

**`no NVIDIA driver visible inside the container`**: `--nv` is missing, or the host has no driver.
`DARTS_ALLOW_NO_GPU=1` continues anyway on CPU engines.

**`ImportError`, or a version mismatch on packages you never installed**: Apptainer bind-mounts `$HOME`, so
`~/.local/lib/python3.*/site-packages` would normally shadow the image. The image sets `PYTHONNOUSERSITE=1` to stop
that; if you override it, this comes back. `--no-home` is the blunt instrument.

**First GPU run is slow, later ones are not**: PTX JIT for an architecture the image carries no cubin for. The result
is cached in `~/.nv/ComputeCache`. Build an image with your own `--cuda-arch` to avoid it entirely (section
*Building the runtime image* of [GPU container images in CI/CD](../for_developers/ci_cd_gpu_images.md)).

**`cannot write to ...` from numba or matplotlib**: the SIF is read-only. The image points `NUMBA_CACHE_DIR` and
`MPLCONFIGDIR` at `$TMPDIR`; set `TMPDIR` to something writable on your node if `/tmp` is not.
