"""
GPU CI preflight (helper_scripts/gpu_ci_preflight.py): a job with --claim keeps its
GPU to itself, other jobs honour the claim without writing anything. No GPU needed.

Run:  python -m pytest tests/test_gpu_ci_preflight.py -q
"""

import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="CI GPU runners are Linux"
)

_SCRIPT = Path(__file__).resolve().parents[1] / "helper_scripts" / "gpu_ci_preflight.py"
_spec = importlib.util.spec_from_file_location("gpu_ci_preflight", _SCRIPT)
preflight = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = preflight  # dataclasses resolve annotations through it
_spec.loader.exec_module(preflight)

UUIDS = [f"GPU-{index}0000000-0000-0000-0000-000000000000" for index in range(4)]


def _summaries():
    """Four idle 32 GiB GPUs."""
    summaries = []
    for index, uuid in enumerate(UUIDS):
        summary = preflight.GpuSummary(index=index, uuid=uuid)
        summary.add(preflight.GpuSample(index, uuid, 0.0, 0, 32768))
        summaries.append(summary)
    return summaries


def _job():
    """A process group standing in for a running CI job step."""
    return subprocess.Popen(["sleep", "60"], start_new_session=True)


def _end(job):
    os.killpg(job.pid, signal.SIGKILL)
    job.wait()


@pytest.fixture
def job_step(monkeypatch):
    """This test as a CI job step with a process group of its own: in a container the
    test runner may be in process group 1, which no claim names."""
    job = _job()
    monkeypatch.setattr(os, "getpgrp", lambda: job.pid)
    yield job.pid
    _end(job)


def test_claimed_gpus_are_avoided_however_idle():
    assert preflight.select_gpu(_summaries(), "", 4096).index == 0
    assert preflight.select_gpu(_summaries(), "", 4096, {0, 1}).index == 2
    assert preflight.select_gpu(_summaries(), "", 4096, {0, 1, 2, 3}).index == 0
    # claimants count down from the other end
    assert preflight.select_gpu(_summaries(), "", 4096, highest=True).index == 3
    assert preflight.select_gpu(_summaries(), "", 4096, {3}, highest=True).index == 2


def test_claim_lives_with_the_job_and_readers_never_write(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "CLAIM_DIR", tmp_path)
    assert preflight.claimed_gpus() == set()  # no claim directory: nothing to honour
    job = _job()
    claim = tmp_path / f"{UUIDS[1]}.{job.pid}"
    claim.touch()
    (tmp_path / f"{UUIDS[2]}.1").touch()  # process group 1 identifies no job
    assert preflight.claimed_gpus() == {UUIDS[1]}
    _end(job)
    assert preflight.claimed_gpus() == set() and claim.exists()  # a reader keeps it
    assert preflight.claimed_gpus(prune=True) == set() and not any(tmp_path.iterdir())
    monkeypatch.setattr(os, "getpgrp", lambda: 1)
    with pytest.raises(OSError):  # so a job there does not claim
        preflight.claim_gpu(_summaries(), _summaries()[0], set(), 4096, [], 0)


def test_claimant_moves_to_a_free_gpu_never_back(
    tmp_path, monkeypatch, capsys, job_step
):
    monkeypatch.setattr(preflight, "CLAIM_DIR", tmp_path)
    summaries, other = _summaries(), _job()
    (tmp_path / f"{UUIDS[1]}.{other.pid}").touch()
    # pid 6 ran on GPU 3 before the decision; jobs then start on each GPU it takes
    polls = iter([[(UUIDS[3], 6), (UUIDS[3], 7)], [(UUIDS[2], 8)], [(UUIDS[0], 9)]])
    monkeypatch.setattr(preflight, "compute_apps", lambda: next(polls))
    try:
        selected = preflight.claim_gpu(
            summaries, summaries[3], set(), 4096, [(UUIDS[3], 6)], 1.0
        )
    finally:
        _end(other)
    log = capsys.readouterr().out
    assert "pid [7] started on GPU 3; moving to GPU 2" in log  # skips claimed GPU 1
    assert "pid [8] started on GPU 2; moving to GPU 0" in log
    assert "pid [9] started on GPU 0; no other GPU is free" in log
    assert selected.index == 0 and (tmp_path / f"{UUIDS[0]}.{job_step}").exists()


def test_claimants_read_and_claim_one_at_a_time(tmp_path, monkeypatch, job_step):
    monkeypatch.setattr(preflight, "CLAIM_DIR", tmp_path)
    summaries, other, locked = _summaries(), _job(), threading.Event()

    def claim_gpu_3():
        """Another claimant, which holds the lock while it claims GPU 3."""
        with preflight.claim_lock():
            locked.set()
            time.sleep(0.2)
            (tmp_path / f"{UUIDS[3]}.{other.pid}").touch()

    thread = threading.Thread(target=claim_gpu_3)
    thread.start()
    locked.wait()
    try:
        selected = preflight.claim_gpu(summaries, summaries[3], set(), 4096, [], 0)
    finally:
        thread.join()
        _end(other)
    assert selected.index == 2


def _preflight(tmp_path, name, claim):
    """Run the script in a new process group that outlives it, like a CI job step."""
    bin_dir = tmp_path / "bin"
    if not bin_dir.exists():  # an nvidia-smi reporting four idle GPUs, no processes
        bin_dir.mkdir()
        rows = "\\n".join(f"{i}, {uuid}, 0, 0, 32768" for i, uuid in enumerate(UUIDS))
        (bin_dir / "nvidia-smi").write_text(
            f'#!/bin/sh\ncase "$1" in --query-gpu=*) printf "{rows}\\n" ;; esac\n'
        )
        (bin_dir / "nvidia-smi").chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE")
    }
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        DARTS_GPU_CLAIM="1" if claim else "0",
        DARTS_GPU_CLAIM_DIR=str(tmp_path / "claims"),
    )
    env_file = tmp_path / f"{name}.env"
    command = (
        f'"{sys.executable}" "{_SCRIPT}" --env-file "{env_file}" --samples 1 '
        "--interval 0 --claim-settle 0 && touch done && exec sleep 60"
    )
    job = subprocess.Popen(
        ["/bin/sh", "-c", command],
        env=env,
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
    )
    for _ in range(300):
        if (tmp_path / "done").exists() or job.poll() is not None:
            break
        subprocess.run(["sleep", "0.05"])
    (tmp_path / "done").unlink()
    return job, env_file.read_text()


def test_end_to_end_benchmark_and_ordinary_jobs(tmp_path):
    bench, env_bench = _preflight(tmp_path, "bench", claim=True)
    try:
        ordinary, env_ordinary = _preflight(tmp_path, "ordinary", claim=False)
        second, env_second = _preflight(tmp_path, "second", claim=True)
        _end(ordinary)
        _end(second)
        assert "CUDA_VISIBLE_DEVICES=3" in env_bench  # claimants count down
        assert "CUDA_VISIBLE_DEVICES=0" in env_ordinary  # other jobs count up
        assert "CUDA_VISIBLE_DEVICES=2" in env_second  # honours the claim
        assert len(list((tmp_path / "claims").iterdir())) == 2  # ordinary wrote none
    finally:
        _end(bench)
    later, env_later = _preflight(tmp_path, "later", claim=True)
    _end(later)
    assert "CUDA_VISIBLE_DEVICES=3" in env_later  # the earlier claims have ended
