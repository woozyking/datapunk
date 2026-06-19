"""
Datapunk worker — executes ONE engine's analytics function in an isolated
subprocess so that engines never share interpreter state, allocator arenas, or
import footprint.

Invoked as:  python -m datapunk.worker <payload.pkl> <result.json>

The payload is a cloudpickled dict:
    {"fn": callable, "args": tuple, "kwargs": dict,
     "iterations": int, "warmup": int, "target_cols": list[str]}

On success it writes a small JSON result (timings + metadata + fingerprint +
peak RSS). It never ships the result frame back. If the parent's memory watchdog
kills this process, no file is written and the parent records an OOM.
"""

from __future__ import annotations

import json
import resource
import sys
import time

import cloudpickle

from datapunk import config
from datapunk.fingerprint import close_result, fingerprint


def _maxrss_mb() -> float:
    """Peak RSS of this process. ru_maxrss is bytes on macOS, KiB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (config.LARGE_CAP_MB * config.LARGE_CAP_MB)
    return rss / config.LARGE_CAP_MB


def main(payload_path: str, result_path: str) -> None:
    with open(payload_path, "rb") as f:
        payload = cloudpickle.load(f)

    fn = payload["fn"]
    args = payload.get("args", ())
    kwargs = payload.get("kwargs", {})
    iterations = payload.get("iterations", 5)
    warmup = payload.get("warmup", 1)
    target_cols = payload["target_cols"]

    result = {"status": "ok"}
    try:
        # Warm-up runs (engine-internal caches, JIT, thread pools) — discarded.
        for _ in range(warmup):
            fn(*args, **kwargs)

        times = []
        last_df, meta = None, {}
        for _ in range(iterations):
            t0 = time.perf_counter()
            raw = fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
                last_df, meta = raw
            else:
                last_df, meta = raw, {}

        # Fingerprint AFTER timing so it never inflates the measured cost.
        result["fingerprint"] = fingerprint(last_df, target_cols)
        close_result(last_df)
        result["times"] = times
        result["meta"] = meta
    except MemoryError:
        result = {"status": "oom_internal"}
    except Exception as exc:  # engine/API error — report cleanly
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    result["maxrss_mb"] = _maxrss_mb()
    with open(result_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
