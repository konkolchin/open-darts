#!/usr/bin/env python3
"""
Summarize shared NVIDIA GPU occupancy and select a device for a CI job.

A benchmark job (the testbench GPU job) passes --claim / DARTS_GPU_CLAIM=1 to keep
its GPU to itself: under a lock on the claim directory it takes the least-loaded
unclaimed GPU, counting down from the highest index (other jobs count up), and
leaves a claim file, which every other preflight on the host honours; it moves if
a job that chose the same GPU just before starts on it.
Without it (two jobs sampling at once see the same idle GPU) the testbench SPE10
run shared a V100 with the model suite and took twice as long. Jobs without
--claim only read claims and never write anything.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import os
import pwd
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

GPU_SAMPLE_QUERY = "index,uuid,utilization.gpu,memory.used,memory.total"
COMPUTE_APP_QUERY = "gpu_uuid,pid"
# One empty file <gpu-uuid>.<pgid> per claiming job step, live while that process
# group runs. The host /tmp is shared by all jobs, also inside Apptainer images.
CLAIM_DIR = Path(os.environ.get("DARTS_GPU_CLAIM_DIR") or "/tmp/open-darts-gpu-claims")
CLAIM_MAX_AGE = 12 * 3600.0  # s, beyond every GPU job timeout


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuSample:
    index: int
    uuid: str
    utilization: float
    memory_used: int
    memory_total: int


@dataclass
class GpuSummary:
    index: int
    uuid: str
    utilization_sum: float = 0.0
    peak_utilization: float = 0.0
    max_memory_used: int = 0
    memory_total: int = 0
    min_memory_free: int | None = None
    sample_count: int = 0

    @property
    def average_utilization(self) -> float:
        return self.utilization_sum / self.sample_count

    def add(self, sample: GpuSample) -> None:
        self.utilization_sum += sample.utilization
        self.peak_utilization = max(self.peak_utilization, sample.utilization)
        self.max_memory_used = max(self.max_memory_used, sample.memory_used)
        self.memory_total = sample.memory_total
        memory_free = sample.memory_total - sample.memory_used
        if self.min_memory_free is None:
            self.min_memory_free = memory_free
        else:
            self.min_memory_free = min(self.min_memory_free, memory_free)
        self.sample_count += 1


@dataclass
class ProcessSummary:
    count: int = 0
    owners: set[str] = field(default_factory=set)


def parse_samples(output: str) -> list[GpuSample]:
    samples = []
    for row in csv.reader(output.splitlines()):
        fields = [value.strip() for value in row]
        if not fields:
            continue
        if len(fields) != 5:
            raise PreflightError(f"Unexpected nvidia-smi sample row: {row!r}")
        try:
            samples.append(
                GpuSample(
                    index=int(fields[0]),
                    uuid=fields[1],
                    utilization=float(fields[2]),
                    memory_used=int(fields[3]),
                    memory_total=int(fields[4]),
                )
            )
        except ValueError as error:
            raise PreflightError(
                f"Non-numeric value in nvidia-smi sample row: {row!r}"
            ) from error
    if not samples:
        raise PreflightError("No NVIDIA GPU was discovered")
    return samples


def collect_samples(count: int, interval: float) -> list[GpuSample]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_SAMPLE_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    print(f"Sampling {count} times at {interval:g}-second intervals...")
    samples = []
    for sample_number in range(count):
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise PreflightError(f"nvidia-smi sampling failed: {detail}")
        samples.extend(parse_samples(result.stdout))
        if sample_number + 1 < count:
            time.sleep(interval)
    return samples


def summarize_samples(samples: list[GpuSample]) -> list[GpuSummary]:
    summaries: dict[int, GpuSummary] = {}
    for sample in samples:
        summary = summaries.setdefault(
            sample.index, GpuSummary(index=sample.index, uuid=sample.uuid)
        )
        summary.add(sample)
    return [summaries[index] for index in sorted(summaries)]


def select_gpu(
    summaries: list[GpuSummary],
    requested_gpu: str,
    min_free_memory: int,
    avoid: set[int] = frozenset(),
    highest: bool = False,
) -> GpuSummary:
    """
    Choose the GPU of a job: GPU_DEVICE if given, else the least-loaded GPU with
    enough free memory that no other job claimed.

    :param summaries: sampled GPUs
    :type summaries: list[GpuSummary]
    :param requested_gpu: physical index from GPU_DEVICE, or '' to choose
    :type requested_gpu: str
    :param min_free_memory: MiB a GPU must have free to be preferred
    :type min_free_memory: int
    :param avoid: indices taken by other jobs, chosen only if nothing else is left
    :type avoid: set[int]
    :param highest: on equal load take the highest index (claimants), else the lowest
    :type highest: bool
    :returns: the selected GPU
    :rtype: GpuSummary
    """
    if requested_gpu:
        try:
            requested_index = int(requested_gpu)
        except ValueError as error:
            raise PreflightError(
                f"Requested GPU_DEVICE={requested_gpu!r} is not a physical GPU index"
            ) from error
        for summary in summaries:
            if summary.index == requested_index:
                return summary
        raise PreflightError(
            f"Requested GPU_DEVICE={requested_index} does not identify an available GPU"
        )

    eligible = [
        summary
        for summary in summaries
        if summary.min_memory_free is not None
        and summary.min_memory_free >= min_free_memory
    ]
    if not eligible:
        print(
            f"WARNING: no GPU has {min_free_memory} MiB free; "
            "selecting the least-loaded device",
        )
        eligible = summaries

    return min(
        eligible,
        key=lambda summary: (
            summary.index in avoid,  # claimed by another job, however idle
            summary.average_utilization,
            summary.peak_utilization,
            summary.max_memory_used,
            -summary.index if highest else summary.index,
        ),
    )


def query_process_owners(pids: set[int]) -> dict[int, str]:
    """
    Resolve the owning user of each of the given process ids.

    Reads /proc rather than shelling out to `ps`: this script also runs inside
    the slim open-DARTS runtime image, which ships no procps-ng, and Apptainer
    gives the container no PID namespace of its own -- so /proc is the host's
    and the pids nvidia-smi reports resolve there exactly as they do outside.

    :param pids: process ids reported by nvidia-smi
    :type pids: set[int]
    :returns: the owner of each pid, or ``uid:<n>`` where the name cannot be
        resolved. Read from inside the image this is a coarser answer than on
        the host: a user namespace reports every uid it does not map as the
        overflow user, so another account's job reads as ``nobody`` rather than
        by name. Only the occupancy report uses this; device selection does
        not. A pid that exited since the nvidia-smi query is omitted.
    :rtype: dict[int, str]
    """
    owners: dict[int, str] = {}
    for pid in pids:
        try:
            uid = os.stat(f"/proc/{pid}").st_uid
        except OSError:
            continue
        try:
            owners[pid] = pwd.getpwuid(uid).pw_name
        except KeyError:
            owners[pid] = f"uid:{uid}"
    return owners


def compute_apps() -> list[tuple[str, int]] | None:
    """(GPU UUID, pid) of every CUDA process, or None if nvidia-smi fails."""
    command = [
        "nvidia-smi",
        f"--query-compute-apps={COMPUTE_APP_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        return None
    processes = []
    for row in csv.reader(result.stdout.splitlines()):
        fields = [value.strip() for value in row]
        if len(fields) == 2 and fields[1].isdigit():
            processes.append((fields[0], int(fields[1])))
    return processes


def claimed_gpus(prune: bool = False) -> set[str]:
    """
    UUIDs of the GPUs claimed by running jobs.

    :param prune: delete the claims of jobs that have ended
    :type prune: bool
    :returns: the UUIDs of the claimed GPUs
    :rtype: set[str]
    """
    claimed = set()
    for path in CLAIM_DIR.glob("GPU-*.*"):
        uuid, _, pgid = path.name.rpartition(".")
        try:
            if int(pgid) <= 1:  # killpg would test this job's group or every process
                raise ProcessLookupError(pgid)
            os.killpg(int(pgid), 0)  # the claiming job step still runs
            live = time.time() - path.stat().st_mtime < CLAIM_MAX_AGE
        except PermissionError:  # it does, as another user
            live = True
        except (OSError, ValueError):
            live = False
        if live:
            claimed.add(uuid)
        elif prune:
            with contextlib.suppress(OSError):
                path.unlink()
    return claimed


@contextlib.contextmanager
def claim_lock() -> Iterator[None]:
    """
    Lock the claim directory, so that claimants read the claims and write theirs
    one at a time; the kernel releases the lock if the holder dies.

    :returns: a context that holds the lock
    :rtype: Iterator[None]
    """
    CLAIM_DIR.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(CLAIM_DIR, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def claim_gpu(
    summaries: list[GpuSummary],
    selected: GpuSummary,
    avoid: set[int],
    min_free_memory: int,
    apps: list[tuple[str, int]],
    settle: float,
) -> GpuSummary:
    """
    Claim a GPU for this job step and watch it for settle seconds.

    A job that chose the same GPU just before, without claiming, shows up only
    when its first CUDA process starts. The claim then moves to a GPU without
    CUDA processes and claims, never back, and the watch starts over there. With
    no such GPU (a single GPU, or all in use) the claim stays and the GPU is shared.

    :param summaries: sampled GPUs
    :type summaries: list[GpuSummary]
    :param selected: the GPU chosen from the samples
    :type selected: GpuSummary
    :param avoid: indices claimed by other jobs, brought up to date under the lock
    :type avoid: set[int]
    :param min_free_memory: MiB a GPU must have free to be preferred
    :type min_free_memory: int
    :param apps: (GPU UUID, pid) of the CUDA processes when the GPU was chosen
    :type apps: list[tuple[str, int]]
    :param settle: seconds to watch the claimed GPU
    :type settle: float
    :returns: the GPU the job uses
    :rtype: GpuSummary
    :raises OSError: if the claim cannot be written
    """
    pgid = os.getpgrp()
    if pgid <= 1:  # e.g. a container's init: not a job step other jobs can check
        raise OSError(f"process group {pgid} does not identify this job")
    with claim_lock():
        avoid |= {s.index for s in summaries if s.uuid in claimed_gpus(prune=True)}
        if selected.index in avoid:  # claimed while this job sampled
            selected = select_gpu(summaries, "", min_free_memory, avoid, highest=True)
        claim = CLAIM_DIR / f"{selected.uuid}.{pgid}"
        claim.touch()
    print(f"Claimed GPU {selected.index} ({claim}); watching it for {settle:g} s")
    known = {pid for uuid, pid in apps if uuid == selected.uuid}
    left = set()
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        time.sleep(min(0.25, settle))  # a CUDA self-test may take under a second
        apps = compute_apps() or []
        new = {pid for uuid, pid in apps if uuid == selected.uuid} - known
        if not new:
            continue
        left.add(selected.index)
        with claim_lock():
            busy = claimed_gpus() | {uuid for uuid, _ in apps}  # claimed or computing
            taken = left | {s.index for s in summaries if s.uuid in busy}
            moved = select_gpu(summaries, "", min_free_memory, taken, highest=True)
            if moved.index not in taken:
                claim = claim.rename(CLAIM_DIR / f"{moved.uuid}.{pgid}")
        if moved.index in taken:
            print(
                f"WARNING: pid {sorted(new)} started on GPU {selected.index}; "
                "no other GPU is free"
            )
            return selected
        print(
            f"WARNING: pid {sorted(new)} started on GPU {selected.index}; "
            f"moving to GPU {moved.index}"
        )
        avoid.discard(moved.index)  # unclaimed now, whatever it was at the start
        selected, known = moved, set()
        deadline = time.monotonic() + settle
    return selected


def query_processes(
    summaries: list[GpuSummary], processes: list[tuple[str, int]] | None
) -> dict[int, ProcessSummary]:
    if processes is None:
        print("WARNING: could not query active CUDA processes")
        return {}

    owners = query_process_owners({pid for _, pid in processes})
    index_by_uuid = {summary.uuid: summary.index for summary in summaries}
    process_summaries: dict[int, ProcessSummary] = {}
    for uuid, pid in processes:
        if uuid not in index_by_uuid:
            continue
        process_summary = process_summaries.setdefault(
            index_by_uuid[uuid], ProcessSummary()
        )
        process_summary.count += 1
        process_summary.owners.add(owners.get(pid, "unknown"))
    return process_summaries


def occupancy_status(
    summary: GpuSummary,
    processes: ProcessSummary,
    active_threshold: float,
    busy_threshold: float,
) -> str:
    if summary.average_utilization >= busy_threshold:
        return "BUSY"
    if summary.average_utilization >= active_threshold:
        return "ACTIVE"
    if processes.count or summary.max_memory_used >= 1024:
        return "RESERVED"
    return "IDLE"


def find_visible_gpu(
    summaries: list[GpuSummary], visible_devices: str
) -> GpuSummary | None:
    if "," in visible_devices:
        return None
    for summary in summaries:
        if visible_devices in {str(summary.index), summary.uuid}:
            return summary
    return None


def print_summary(
    summaries: list[GpuSummary],
    process_summaries: dict[int, ProcessSummary],
    selected_index: int | None,
    active_threshold: float,
    busy_threshold: float,
) -> None:
    print("GPU occupancy (* = selected/preallocated physical GPU):")
    print("    GPU  STATE       AVG%  PEAK%     MEMORY GiB  PROCS  OWNERS")
    for summary in summaries:
        processes = process_summaries.get(summary.index, ProcessSummary())
        status = occupancy_status(summary, processes, active_threshold, busy_threshold)
        owner_names = sorted(processes.owners)
        owners = ",".join(owner_names[:4]) or "-"
        if len(owner_names) > 4:
            owners += f",+{len(owner_names) - 4} more"
        marker = "*" if summary.index == selected_index else " "
        memory = (
            f"{summary.max_memory_used / 1024:.1f}/{summary.memory_total / 1024:.1f}"
        )
        print(
            f" {marker}  {summary.index:>3}  {status:<9} "
            f"{summary.average_utilization:>5.1f} "
            f"{summary.peak_utilization:>6.1f} "
            f"{memory:>14} {processes.count:>6}  {owners}"
        )
    print(
        f"States: BUSY >= {busy_threshold:g}%; ACTIVE >= {active_threshold:g}%; "
        "RESERVED = lower load but holding memory/processes; otherwise IDLE."
    )


def write_environment(path: Path, values: dict[str, str]) -> None:
    content = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in values.items()
    )
    path.write_text(content, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize NVIDIA GPU occupancy and select a CI device."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="write shell exports for the calling CI job to this file",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--active-threshold", type=float, default=10.0)
    parser.add_argument("--busy-threshold", type=float, default=80.0)
    parser.add_argument(
        "--claim",
        action="store_true",
        default=os.environ.get("DARTS_GPU_CLAIM") == "1",
        help="claim the GPU for this job, for benchmarks (default: DARTS_GPU_CLAIM=1)",
    )
    parser.add_argument("--claim-settle", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_arguments()
    if args.samples < 1 or args.interval < 0:
        raise PreflightError("Sample count must be positive and interval non-negative")
    if not 0 <= args.active_threshold < args.busy_threshold <= 100:
        raise PreflightError(
            "Occupancy thresholds must satisfy 0 <= active < busy <= 100"
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    requested_gpu = os.environ.get("GPU_DEVICE", "").strip()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"=== GPU CI preflight | {socket.getfqdn()} | {timestamp} ===")
    print(
        f"Initial allocation: CUDA_VISIBLE_DEVICES={visible_devices or '<unset>'}; "
        f"GPU_DEVICE={requested_gpu or '<unset>'}"
    )

    samples = collect_samples(args.samples, args.interval)
    summaries = summarize_samples(samples)
    apps = compute_apps()
    process_summaries = query_processes(summaries, apps)
    claimed = claimed_gpus(prune=args.claim)
    avoid = {summary.index for summary in summaries if summary.uuid in claimed}

    if visible_devices:
        selected = find_visible_gpu(summaries, visible_devices)
        environment = {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "GPU_DEVICE": requested_gpu or "0",
        }
        decision = f"preserve preallocated CUDA_VISIBLE_DEVICES={visible_devices}"
    else:
        selected = select_gpu(
            summaries, requested_gpu, args.min_free_mib, avoid, highest=args.claim
        )
        reason = "explicit GPU_DEVICE override" if requested_gpu else "lowest load"
        if avoid:
            reason += f", GPU {sorted(avoid)} claimed by other CI jobs"
        if args.claim and not requested_gpu:
            try:
                selected = claim_gpu(
                    summaries,
                    selected,
                    avoid,
                    args.min_free_mib,
                    apps or [],
                    args.claim_settle,
                )
            except OSError as error:
                print(f"WARNING: could not claim GPU {selected.index}: {error}")
        environment = {
            "DARTS_CI_PHYSICAL_GPU_DEVICE": str(selected.index),
            "CUDA_VISIBLE_DEVICES": str(selected.index),
            "GPU_DEVICE": "0",
        }
        decision = f"physical GPU {selected.index} ({reason}) -> logical CUDA device 0"

    selected_index = selected.index if selected is not None else None
    print_summary(
        summaries,
        process_summaries,
        selected_index,
        args.active_threshold,
        args.busy_threshold,
    )
    print(f"Decision: {decision}")
    if selected is not None and selected.index in avoid:
        print(f"WARNING: GPU {selected.index} is shared with a claiming CI job")
    if selected is not None and selected.average_utilization >= args.busy_threshold:
        print(
            f"WARNING: selected GPU {selected.index} is BUSY at "
            f"{selected.average_utilization:.1f}% average utilization",
        )

    write_environment(args.env_file, environment)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
