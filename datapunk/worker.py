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

import gc
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
    bytes_per_mb = getattr(config, "BYTES_PER_MB", 1024 * 1024)
    kib_per_mb = getattr(config, "KIB_PER_MB", 1024)
    if sys.platform == "darwin":
        return rss / bytes_per_mb
    return rss / kib_per_mb


def _unpack(raw):
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
        return raw
    return raw, {}


def _dispose(raw) -> None:
    df, _ = _unpack(raw)
    close_result(df)
    del df


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
    last_df = None
    try:
        # Warm-up runs (engine-internal caches, JIT, thread pools) — discarded.
        for _ in range(warmup):
            raw = fn(*args, **kwargs)
            _dispose(raw)
            del raw
            gc.collect()

        times = []
        meta = {}
        for i in range(iterations):
            t0 = time.perf_counter()
            raw = fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
            df, meta = _unpack(raw)

            # Keep only the final successful output for post-timing verification.
            # Earlier iterations are closed immediately so repeated large runs do
            # not accumulate native handles or Python objects in the worker.
            if i == iterations - 1:
                last_df = df
            else:
                close_result(df)
                del raw, df
                gc.collect()

        # Fingerprint AFTER timing so it never inflates the measured cost.
        result["fingerprint"] = fingerprint(last_df, target_cols)
        result["times"] = times
        result["meta"] = meta
    except MemoryError:
        result = {"status": "oom_internal"}
    except Exception as exc:  # engine/API error — report cleanly
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if last_df is not None:
            close_result(last_df)
        gc.collect()

    result["maxrss_mb"] = _maxrss_mb()
    with open(result_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
