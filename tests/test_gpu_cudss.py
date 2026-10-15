"""
The cuDSS GPU direct solver (CuDSSSolverSpec) of a build that bundles libcudss, such as the
GPU runtime image (test-apptainer-runtime-gpu). Skipped without cuDSS or without a GPU.

Run:  python -m pytest tests/test_gpu_cudss.py -q
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

import darts.linear_solvers
from darts.linear_solvers import CuDSSSolverSpec

_CUDSS = Path(darts.linear_solvers.__file__).parent / "libcudss.so.0"
_MODEL = Path(__file__).resolve().parents[1] / "models" / "2ph_do" / "model.py"


def _gpu_visible():
    """Whether nvidia-smi lists a GPU."""
    smi = shutil.which("nvidia-smi")
    return (
        bool(smi) and subprocess.run([smi, "-L"], capture_output=True).returncode == 0
    )


pytestmark = pytest.mark.skipif(
    not (_CUDSS.exists() and _gpu_visible()),
    reason="needs a build with cuDSS and a GPU",
)


def test_cudss_solves_every_newton_step_directly(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("model_2ph_do", _MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)  # set_output() writes to ./output
    model = module.Model()
    model.linear_solver.spec = CuDSSSolverSpec()
    model.init(platform="gpu")
    model.set_output()
    model.run(days=30)
    stats = model.nonlinear_solver.stats
    assert stats.n_newton_total > 0
    # a direct solve: one linear iteration per Newton iteration
    assert stats.n_linear_total == stats.n_newton_total
