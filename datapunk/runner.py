"""
Datapunk runner — spawns an isolated worker per engine and enforces a *physical*
(RSS) memory ceiling from the parent via a high-frequency watchdog.

Why a watchdog instead of ``resource.RLIMIT_AS``:
``RLIMIT_AS`` caps virtual address space, which multi-threaded, arena-allocating
engines (Polars, DuckDB, Daft) reserve far in excess of the RAM they actually
commit — so it fires on reservation strategy, not memory pressure, and behaves
differently per engine and per core count. Polling RSS and SIGKILLing on breach
measures and enforces real memory, works identically on macOS and Linux, and —
crucially — turns OOM into a recorded *result* instead of a crashed kernel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Tuple

import cloudpickle
import psutil

from datapunk import config


def run_engine(
    fn: Callable,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    *,
    iterations: int = 5,
    warmup: int = 1,
    target_cols: List[str],
    limit_mb: int | None = None,
    poll: float = 0.005,
    env: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Run *fn* in a fresh subprocess and return a result dict:

        {"status": "ok"|"oom"|"error",
         "time": <avg seconds, ok only>,
         "times": [...],
         "peak_mb": <peak RSS observed>,
         "meta": {...}, "fingerprint": {...},
         "error": <str, error only>}

    If *limit_mb* is set and the worker's RSS exceeds it, the worker is killed
    and status is ``"oom"``.
    """
    kwargs = kwargs or {}
    with tempfile.TemporaryDirectory(prefix="datapunk_") as td:
        payload_path = os.path.join(td, "payload.pkl")
        result_path = os.path.join(td, "result.json")
        with open(payload_path, "wb") as f:
            cloudpickle.dump(
                {
                    "fn": fn,
                    "args": args,
                    "kwargs": kwargs,
                    "iterations": iterations,
                    "warmup": warmup,
                    "target_cols": target_cols,
                },
                f,
            )

        cmd = [sys.executable, "-m", "datapunk.worker", payload_path, result_path]
        proc_env = {**os.environ, **(env or {})}
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, env=proc_env)
        ps = psutil.Process(proc.pid)
        limit_bytes = (limit_mb or 0) * config.LARGE_CAP_MB * config.LARGE_CAP_MB
        peak = 0
        oomed = False

        while proc.poll() is None:
            try:
                rss = _tree_rss(ps)
            except psutil.NoSuchProcess:
                break
            if rss > peak:
                peak = rss
            if limit_mb is not None and rss > limit_bytes:
                _kill_tree(ps)
                oomed = True
                break
            time.sleep(poll)

        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        proc.wait()

        peak_mb = peak / (config.LARGE_CAP_MB * config.LARGE_CAP_MB)

        if oomed:
            return {"status": "oom", "peak_mb": peak_mb, "limit_mb": limit_mb}

        if not os.path.exists(result_path):
            # Killed by the OS OOM killer, or crashed before writing.
            return {
                "status": "error",
                "peak_mb": peak_mb,
                "error": stderr.strip()[-500:] or "worker produced no result",
            }

        with open(result_path) as f:
            res = json.load(f)

        # Worker's own peak (ru_maxrss) is exact; watchdog can miss a spike
        # between samples. Report the larger.
        peak_mb = max(peak_mb, res.get("maxrss_mb", 0.0))

        if res["status"] == "ok":
            times = res["times"]
            return {
                "status": "ok",
                "time": sum(times) / len(times),
                "times": times,
                "peak_mb": peak_mb,
                "meta": res.get("meta", {}),
                "fingerprint": res.get("fingerprint"),
            }
        if res["status"] == "oom_internal":
            return {"status": "oom", "peak_mb": peak_mb, "limit_mb": limit_mb}
        return {
            "status": "error",
            "peak_mb": peak_mb,
            "error": res.get("error", "unknown"),
        }


def _tree_rss(ps: psutil.Process) -> int:
    """Total RSS of the worker and any children (Dask/Daft can spawn helpers)."""
    total = ps.memory_info().rss
    for child in ps.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def _kill_tree(ps: psutil.Process) -> None:
    procs = ps.children(recursive=True) + [ps]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=3)
