#!/usr/bin/env bash
# =============================================================================
# Build the portable open-DARTS GPU runtime image.
#
# Two phases, because the toolchain has no business being in the shipped image:
#
#   1. compile a *portable* wheel inside the builder image (builder-gpu-image.def):
#      multi-architecture cubins plus a PTX fallback instead of CMake's
#      `native`, and an explicit -march baseline instead of `-march=native`
#   2. assemble the runtime image (runtime-gpu-image.def) around that wheel
#
# Phase 1 is skipped when --wheel is given, which is what CI does with the
# artifact from build-linux-tahiti-image-gpu.
#
#   .cicd/build-linux/build_runtime_image.sh                     # both phases
#   .cicd/build-linux/build_runtime_image.sh --wheel dist/*.whl  # assemble only
#   .cicd/build-linux/build_runtime_image.sh --cuda-arch 80 --cpu-arch native
#
# Run it from anywhere; it locates the repository root itself.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# One cubin per compute-capability *major*, because a cubin for X.y also runs
# on X.z when z >= y -- so sm_70 covers Turing, sm_80 covers A10/L40/RTX 40xx,
# and so on. sm_100 (B200/GB200) needs its own entry: compute capability 10.x
# is not binary compatible with 9.x below it, and the compute_120 PTX cannot
# be JIT'd *down* to it either. compute_120 PTX is the fallback for GPUs newer
# than sm_120 (JIT'd on first use, cached under ~/.nv).
CUDA_ARCH_DEFAULT='70-real;80-real;90-real;100-real;120-real;120-virtual'
# x86-64-v3 is AVX2 + FMA + BMI: every server CPU since Haswell (2013). Drop to
# x86-64-v2 for pre-2013 hosts, or `native` when the image never leaves one machine.
CPU_ARCH_DEFAULT='x86-64-v3'
CUDA_VERSION_DEFAULT='12.8.1'
PY_VER_DEFAULT='3.12'
EXTRAS_DEFAULT='test'
# the cuDSS direct solver, from NVIDIA's pip wheel for the image's CUDA major (the CUDA 12
# build spans sm_50 to sm_120); pinned, as its 0.x releases change the API under the same
# soname. Keep in sync with build-linux-tahiti-image-gpu
CUDSS_VERSION_DEFAULT='0.8.0.10'
# nproc honours OMP_NUM_THREADS, which HPC login shells and CI runners routinely
# pin to 1 -- that would silently turn this into a single-threaded build.
JOBS_DEFAULT="$(OMP_NUM_THREADS= nproc 2>/dev/null || echo 8)"

CUDA_ARCH="${CUDA_ARCH:-$CUDA_ARCH_DEFAULT}"
CPU_ARCH="${CPU_ARCH:-$CPU_ARCH_DEFAULT}"
CUDA_VERSION="${CUDA_VERSION:-$CUDA_VERSION_DEFAULT}"
# Remember whether the caller chose one: when they did not, and a prebuilt
# wheel is supplied, the wheel's own ABI tag decides (see below).
PY_VER_EXPLICIT=0
[ -n "${PY_VER:-}" ] && PY_VER_EXPLICIT=1
PY_VER="${PY_VER:-$PY_VER_DEFAULT}"
EXTRAS="${EXTRAS:-$EXTRAS_DEFAULT}"
JOBS="${JOBS:-$JOBS_DEFAULT}"
BUILDER_IMAGE="${BUILDER_IMAGE:-}"
WHEEL=""
OUTPUT=""
CUDSS_VERSION="${CUDSS_VERSION:-$CUDSS_VERSION_DEFAULT}"
WITH_CUDSS=1
KEEP_CONTEXT=0
KEEP_CUDA_COMPAT="${KEEP_CUDA_COMPAT:-1}"   # NVIDIA forward-compat libcuda, 307 MB

usage() {
    cat <<EOF
Usage: ${0##*/} [options]

  --wheel PATH           use this prebuilt wheel instead of compiling one
  --builder-image PATH   builder SIF for phase 1 (built from builder-gpu-image.def if absent)
  --output PATH          output SIF (default: open-darts-gpu-<version>.sif in the repo root)
  --cuda-version VER     CUDA toolkit/runtime version      (default: ${CUDA_VERSION_DEFAULT})
  --cuda-arch LIST       CMake CUDA architectures          (default: ${CUDA_ARCH_DEFAULT})
  --cpu-arch ISA         -march baseline, or 'off'         (default: ${CPU_ARCH_DEFAULT})
  --py-ver VER           Python version                    (default: ${PY_VER_DEFAULT})
  --extras LIST          wheel extras, comma separated     (default: ${EXTRAS_DEFAULT})
  --no-cudss             leave out the cuDSS direct solver (-103 MB of SIF)  (default: cuDSS ${CUDSS_VERSION_DEFAULT})
  --no-cuda-compat       drop NVIDIA's forward-compat libcuda (-307 MB, needs a current driver)
  -j, --jobs N           compile jobs                      (default: ${JOBS_DEFAULT})
  --keep-context         keep the staged wheelhouse for inspection
  -h, --help             this message

Environment variables of the same name override the defaults; flags win over both.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --wheel)          WHEEL="$2"; shift 2 ;;
        --builder-image)  BUILDER_IMAGE="$2"; shift 2 ;;
        --output)         OUTPUT="$2"; shift 2 ;;
        --cuda-version)   CUDA_VERSION="$2"; shift 2 ;;
        --cuda-arch)      CUDA_ARCH="$2"; shift 2 ;;
        --cpu-arch)       CPU_ARCH="$2"; shift 2 ;;
        --py-ver)         PY_VER="$2"; PY_VER_EXPLICIT=1; shift 2 ;;
        --extras)         EXTRAS="$2"; shift 2 ;;
        --with-cudss)     WITH_CUDSS=1; shift ;;   # the default; kept for older command lines
        --no-cudss)       WITH_CUDSS=0; shift ;;
        --no-cuda-compat) KEEP_CUDA_COMPAT=0; shift ;;
        -j|--jobs)        JOBS="$2"; shift 2 ;;
        --keep-context)   KEEP_CONTEXT=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found on PATH" >&2; exit 1; }

# Heavy I/O belongs somewhere with room; /tmp is often a small tmpfs.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/var/tmp}/apptainer-tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${TMPDIR:-/var/tmp}/apptainer-cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

CONTEXT="${REPO_ROOT}/.cicd/build-linux/.context"
WHEELHOUSE="${CONTEXT}/wheelhouse"
cleanup() { [ "$KEEP_CONTEXT" -eq 1 ] || rm -rf "$CONTEXT"; }
trap cleanup EXIT

DARTS_VERSION="$(sed -n 's/^version[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "${REPO_ROOT}/pyproject.toml" | head -n1)"
VCS_REF="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "==============================================================="
echo " open-DARTS runtime image"
echo "   version      ${DARTS_VERSION} (${VCS_REF})"
echo "   CUDA         ${CUDA_VERSION}"
echo "   GPU targets  ${CUDA_ARCH}"
echo "   CPU baseline ${CPU_ARCH}"
echo "   Python       ${PY_VER}   extras [${EXTRAS}]"
echo "==============================================================="

# ---------------------------------------------------------------------------
# Phase 1 -- portable wheel
# ---------------------------------------------------------------------------
if [ -z "$WHEEL" ]; then
    if [ -z "$BUILDER_IMAGE" ]; then
        BUILDER_IMAGE="${APPTAINER_TMPDIR}/open-darts-builder.sif"
        if [ ! -f "$BUILDER_IMAGE" ]; then
            echo "-- building the builder image (one-off, ~20 min) -> ${BUILDER_IMAGE}"
            ( cd "$REPO_ROOT" && apptainer build --fakeroot \
                --build-arg CUDA_VERSION="$CUDA_VERSION" \
                "$BUILDER_IMAGE" .cicd/build-linux/builder-gpu-image.def )
        else
            echo "-- reusing builder image ${BUILDER_IMAGE}"
        fi
    fi
    # -e, not -f: a sandbox builder image is a directory, and exec'ing one is a
    # normal apptainer workflow (and much faster to iterate on).
    [ -e "$BUILDER_IMAGE" ] || { echo "builder image not found: $BUILDER_IMAGE" >&2; exit 1; }

    CUDSS_FLAG='--no-cudss'
    CUDSS_PIP=':'
    if [ "$WITH_CUDSS" -eq 1 ]; then
        CUDSS_FLAG='--cudss'
        # the build finds it in the env's site-packages (thirdparty/thirdparty_cudss.cmake)
        CUDSS_PIP="pip install -q nvidia-cudss-cu${CUDA_VERSION%%.*}==${CUDSS_VERSION}"
    fi

    echo "-- compiling the portable wheel inside the builder image"
    # --nv is deliberately NOT passed: nothing here needs a GPU, and without it
    # the build cannot accidentally depend on the machine's own device.
    # The environment is created at a *prefix* inside the bind-mounted workspace,
    # not as a named env: a SIF is read-only, so `conda create -n` would have to
    # write into /opt/conda in the image. Its packages go to a cache of its own
    # beside it (--env: the image's %environment would override a plain
    # variable), not to the image's default in /tmp, which other builds share.
    # The bash -lc script is one double-quoted string, comments included:
    # a double quote anywhere in it ends the string early.
    mkdir -p "$CONTEXT"
    touch "${CONTEXT}/build-start"
    apptainer exec --bind "${REPO_ROOT}:/work" --env CONDA_PKGS_DIRS=/work/.wheelbuild-pkgs \
        "$BUILDER_IMAGE" bash -lc "
        set -euxo pipefail
        source /opt/conda/etc/profile.d/conda.sh
        # The %post acceptance is written under the build-time HOME, which is
        # not the HOME seen at run time (Apptainer bind-mounts the caller's),
        # so conda asks again here. Idempotent.
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null
        rm -rf /work/.wheelbuild-env /work/.wheelbuild-pkgs
        conda create -p /work/.wheelbuild-env python=${PY_VER} -q -y
        conda activate /work/.wheelbuild-env
        ${CUDSS_PIP}
        cd /work
        unset GSELINSOLVERSPATH
        export OD_CMAKE_ARGS='-D ENABLE_BOS_SOLVERS=OFF'
        export CUDA_ARCH='${CUDA_ARCH}'
        export CPU_ARCH='${CPU_ARCH}'
        # -p builds and bundles IPhreeqc into the wheel, which the chemistry
        # models need. It also conda-installs Reaktoro into this throwaway
        # build env -- a few minutes of solving for something that cannot go
        # into the runtime image (conda-forge only, no PyPI wheel). Unavoidable
        # side effect of the flag; see README.md, 'Known limitation: Reaktoro'.
        ./helper_scripts/build_install_darts_gpu.sh -c -p ${CUDSS_FLAG} -j ${JOBS}
    "
    rm -rf "${REPO_ROOT}/.wheelbuild-env" "${REPO_ROOT}/.wheelbuild-pkgs"
    # Only a wheel this build wrote: in CI, dist/ already holds the native
    # wheel of build-linux-tahiti-image-gpu.
    WHEEL=""
    for w in "${REPO_ROOT}"/dist/*.whl; do
        if [ "$w" -nt "${CONTEXT}/build-start" ] && { [ -z "$WHEEL" ] || [ "$w" -nt "$WHEEL" ]; }; then
            WHEEL="$w"
        fi
    done
    [ -n "$WHEEL" ] || { echo "no wheel produced in ${REPO_ROOT}/dist" >&2; exit 1; }
fi

[ -f "$WHEEL" ] || { echo "wheel not found: $WHEEL" >&2; exit 1; }
echo "-- wheel: $WHEEL ($(du -h --apparent-size "$WHEEL" | cut -f1))"

# The venv inside the image has to match the wheel's ABI tag. A cp311 wheel in a
# cp312 venv is rejected by pip with the distinctly unhelpful
#   ERROR: ...whl is not a supported wheel on this platform.
# CI hits this because the wheel comes from build-linux-tahiti-image-gpu, which
# builds with the project's DEFAULT_PY_VER, while this script defaults to 3.12.
WHEEL_PY="$(basename "$WHEEL" | sed -n 's/.*-cp\([0-9]\)\([0-9][0-9]*\)-cp[0-9][0-9]*-.*/\1.\2/p')"
if [ -n "$WHEEL_PY" ]; then
    if [ "$PY_VER_EXPLICIT" -eq 1 ] && [ "$PY_VER" != "$WHEEL_PY" ]; then
        echo "FATAL: --py-ver ${PY_VER} was requested, but $(basename "$WHEEL") is built" >&2
        echo "       for Python ${WHEEL_PY}. pip would reject it inside the image." >&2
        exit 1
    fi
    if [ "$PY_VER" != "$WHEEL_PY" ]; then
        echo "-- wheel is cp${WHEEL_PY/./}; using Python ${WHEEL_PY} for the image venv"
        PY_VER="$WHEEL_PY"
    fi
fi

# ---------------------------------------------------------------------------
# Phase 2 -- assemble the runtime image
# ---------------------------------------------------------------------------
rm -rf "$CONTEXT"
mkdir -p "$WHEELHOUSE"
cp "$WHEEL" "$WHEELHOUSE/"

[ -n "$OUTPUT" ] || OUTPUT="${REPO_ROOT}/open-darts-gpu-${DARTS_VERSION}.sif"

echo "-- assembling ${OUTPUT}"
# Optional: zstd-19 makes the SIF smaller and faster to decompress than the
# default gzip. Probe the mksquashfs apptainer runs (its bundled copy, not the one
# on PATH); without it, or without zstd (squashfs-tools < 4.4), gzip is used.
MKSQ=""
for c in "$(sed -n 's/^[[:space:]]*mksquashfs path[[:space:]]*=[[:space:]]*//p' \
              /etc/apptainer/apptainer.conf 2>/dev/null | tail -1)" \
         /usr/libexec/apptainer/bin/mksquashfs \
         /usr/libexec/singularity/bin/mksquashfs \
         "$(command -v mksquashfs 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then MKSQ="$c"; break; fi
done
SQUASH_ARGS=()
if [ -z "$MKSQ" ]; then
    echo "   mksquashfs not found; leaving compression to apptainer"
elif "$MKSQ" /dev/null /dev/null -comp zstd -help 2>&1 | grep -qi 'not supported'; then
    echo "   $MKSQ has no zstd; using the default gzip"
else
    echo "   using zstd via $MKSQ"
    SQUASH_ARGS=(--mksquashfs-args '-comp zstd -Xcompression-level 19')
fi
# -T: the runtime image's %test runs a CUDA kernel, which needs --nv and a GPU.
# Assembling and verifying are separate steps -- the verification runs below if
# this machine has a GPU, and in CI it is a job of its own.
( cd "$REPO_ROOT" && apptainer build --fakeroot --force --notest \
    ${SQUASH_ARGS[@]+"${SQUASH_ARGS[@]}"} \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg PY_VER="$PY_VER" \
    --build-arg EXTRAS="$EXTRAS" \
    --build-arg KEEP_CUDA_COMPAT="$KEEP_CUDA_COMPAT" \
    --build-arg CUDSS="$WITH_CUDSS" \
    --build-arg CUDA_ARCH="$CUDA_ARCH" \
    --build-arg CPU_ARCH="$CPU_ARCH" \
    --build-arg DARTS_VERSION="$DARTS_VERSION" \
    --build-arg VCS_REF="$VCS_REF" \
    "$OUTPUT" .cicd/build-linux/runtime-gpu-image.def )

echo
echo "-- built ${OUTPUT} ($(du -h --apparent-size "$OUTPUT" | cut -f1))"

# Verify on the spot when this machine can: the self-test runs a real CUDA
# kernel, which is the only way to know the fatbinary carries code this GPU can
# execute. A build host without a GPU just gets the instructions.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "-- running the image self-test on this host's GPU"
    apptainer test --nv "$OUTPUT"
else
    echo "-- no GPU here; verify on one with: apptainer test --nv ${OUTPUT}"
fi

echo
echo "==============================================================="
echo " built: ${OUTPUT}  ($(du -h --apparent-size "$OUTPUT" | cut -f1))"
echo
echo " verify:  apptainer test --nv ${OUTPUT}"
echo " inspect: apptainer inspect ${OUTPUT}"
echo " run:     apptainer run --nv ${OUTPUT} model.py"
echo "==============================================================="
