"""
DatapunkReporter — orchestrates the two-run benchmark model:

  • SMALL  : single month, no cap        → apples-to-apples speed + correctness
  • LARGE  : full window, RSS cap         → out-of-core behaviour, OOMs surfaced

Each engine runs in an isolated subprocess (see ``runner``). The reporter keeps
the analytics functions exactly as the notebook lays them out — it cloudpickles
them into the worker, so nothing about how you author a suite changes.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import psutil

from datapunk import config, dataio
from datapunk.fingerprint import close_result, compare, to_pandas_frame
from datapunk.runner import run_engine

_BASELINE = "pandas"


def _sort_key(item):
    name = item[0]
    return (_BASELINE not in name.lower(), name.lower())


class DatapunkReporter:
    def __init__(
        self,
        *,
        small_months: int = config.SMALL_MONTHS,
        large_cap_mb: int = config.LARGE_CAP_MB,
        iterations: int = config.ITERATIONS,
        warmup: int = config.WARMUP,
        large_iterations: int | None = None,
        large_warmup: int | None = None,
        with_lookup: bool = False,
    ):
        self.iterations = iterations
        self.warmup = warmup
        self.large_iterations = (
            config.LARGE_ITERATIONS if large_iterations is None else large_iterations
        )
        self.large_warmup = (
            config.LARGE_WARMUP if large_warmup is None else large_warmup
        )
        self.large_cap_mb = large_cap_mb

        # Resolve + download data.
        all_pairs = dataio.monthly_files()
        self._all_paths = dataio.bootstrap(all_pairs)
        self._all_paths = dataio.unify_parquet_schema(self._all_paths)
        self.small_paths = self._all_paths[:small_months]
        self.large_paths = self._all_paths

        self.lookup_path = None
        if with_lookup:
            lk_local, lk_url = dataio.lookup_file()
            self.lookup_path = dataio.bootstrap([(lk_local, lk_url)])[0]

        # Warm the page cache once for the full set (kernel-wide, shared).
        dataio.warm_page_cache(self._all_paths)

        # results[mode][engine_label] = {status, time, peak_mb, ...}
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._fingerprints: Dict[str, Dict[str, Any]] = {}
        self._verify: Dict[str, str] = {}
        self._print_manifest()

    # ------------------------------------------------------------------
    def _print_manifest(self):
        mem = psutil.virtual_memory()
        print("Datapunk Environment")
        print("-" * 52)
        print(f"  OS:        {platform.system()} {platform.release()}")
        print(f"  CPU:       {_cpu_model()} ({os.cpu_count()} cores)")
        print(
            f"  RAM:       {mem.total / getattr(config, 'BYTES_PER_GB', 1024**3):.1f} GB"
        )
        print(f"  Python:    v{platform.python_version()}")
        print(
            f"  Small run: {len(self.small_paths)} month, uncapped, "
            f"{self.iterations} iter + {self.warmup} warmup"
        )
        print(
            f"  Large run: {len(self.large_paths)} months, cap = "
            f"{self.large_cap_mb} MB (physical RSS), "
            f"{self.large_iterations} iter + {self.large_warmup} warmup"
        )
        print("-" * 52)

    # ------------------------------------------------------------------
    def run_all(
        self,
        analytics: List[Callable],
        *,
        target_cols: List[str],
        pass_lookup: bool = False,
    ):
        """Run every engine under both modes; verify cross-engine consistency."""
        lookup_args = (self.lookup_path,) if pass_lookup else ()
        labels: Dict[int, str] = {}  # id(fn) -> label, learned from the small run

        for mode, paths, cap, iterations, warmup in (
            ("small", self.small_paths, None, self.iterations, self.warmup),
            (
                "large",
                self.large_paths,
                self.large_cap_mb,
                self.large_iterations,
                self.large_warmup,
            ),
        ):
            data_arg = list(paths)
            print(
                f"\n▶ {mode.upper()} run "
                f"({len(paths)} month{'s' if len(paths) > 1 else ''}, "
                f"cap={'none' if cap is None else f'{cap} MB'})"
            )

            # Resolve engine names cheaply via cloudpickle round-trip metadata:
            # we run each engine once in-mode and read its returned meta.
            mode_results: Dict[str, Dict[str, Any]] = {}
            pending: List[Tuple[str, Callable]] = []
            for fn in analytics:
                r = run_engine(
                    fn,
                    args=(data_arg, *lookup_args),
                    iterations=iterations,
                    warmup=warmup,
                    target_cols=target_cols,
                    limit_mb=cap,
                )
                meta = r.get("meta", {})
                if meta.get("engine"):
                    name = meta["engine"]
                    ver = meta.get("version", "?")
                    label = f"{name} v{ver}" if ver != "?" else name
                    labels[id(fn)] = label
                else:
                    # OOM/error: no metadata returned — reuse the label learned
                    # from the (uncapped) small run, else the function name.
                    label = labels.get(id(fn), fn.__name__)
                    name = label
                r["_name"] = name
                mode_results[label] = r
                pending.append((label, r))
                status = r["status"]
                detail = (
                    f"{r['time'] * 1000:.0f} ms" if status == "ok" else status.upper()
                )
                print(f"    {label:<22} {detail:>10}  peak {r['peak_mb']:.0f} MB")

            # Order: baseline first, then alphabetical.
            ordered = dict(sorted(mode_results.items(), key=_sort_key))
            self.results[mode] = ordered

            # Verify consistency among engines that succeeded.
            self._verify[mode] = self._verify_mode(ordered)
            print(f"    → consistency: {self._verify[mode]}")

    def _verify_mode(self, ordered: Dict[str, Dict[str, Any]]) -> str:
        ref_label = ref_fp = None
        for label, r in ordered.items():
            if r["status"] != "ok":
                continue
            fp = r.get("fingerprint")
            if fp is None:
                continue
            if ref_fp is None:
                ref_label, ref_fp = label, fp
                continue
            ok, reason = compare(ref_fp, fp)
            if not ok:
                return f"❌ {label} differs from {ref_label} ({reason})"
        n_ok = sum(1 for r in ordered.values() if r["status"] == "ok")
        if ref_fp is None:
            return "⚠ no engine produced a result"
        rows = ref_fp["n_rows"]
        return f"✅ {n_ok} engine(s) agree ({rows:,} rows)"

    # ------------------------------------------------------------------
    def set_analysis_fn(self, fn: Callable, *, mode="large", pass_lookup=False):
        """Designate which engine fn produces the (uncapped) chart data."""
        self._analysis_fn = fn
        self._analysis_mode = mode
        self._analysis_lookup = pass_lookup

    def get_analysis_df(self, target_cols: List[str]) -> pd.DataFrame:
        """Run the designated analysis fn uncapped and return a pandas frame."""
        paths = self.small_paths if self._analysis_mode == "small" else self.large_paths
        args = (list(paths),)
        if self._analysis_lookup:
            args += (self.lookup_path,)
        raw = self._analysis_fn(*args)
        try:
            return to_pandas_frame(raw, target_cols=target_cols)
        finally:
            close_result(raw)

    # ------------------------------------------------------------------
    def show_scorecard(self, title: str):
        from IPython.display import Markdown, display

        for mode in ("small", "large"):
            if mode not in self.results:
                continue
            res = self.results[mode]
            ok_times = {k: v["time"] for k, v in res.items() if v["status"] == "ok"}
            max_t = max(ok_times.values()) if ok_times else 1.0
            best = (
                max(ok_times, key=lambda k: max_t / ok_times[k]) if ok_times else None
            )

            cap = self.large_cap_mb if mode == "large" else None
            rows = []
            for label, r in res.items():
                if r["status"] == "ok":
                    vf = max_t / r["time"]
                    vf_s = f"**{vf:.1f}×**" if label == best else f"{vf:.1f}×"
                    t_s = f"{r['time']:.4f}s"
                    stress = f"{r['peak_mb'] / cap:.2f}" if cap else "—"
                    peak = f"{r['peak_mb']:.0f} MB"
                elif r["status"] == "oom":
                    vf_s, t_s, peak = "—", "**OOM**", f"≥{r['peak_mb']:.0f} MB"
                    stress = f"≥{r['peak_mb'] / cap:.2f}" if cap else "—"
                else:
                    vf_s, t_s, peak, stress = "—", "ERROR", "—", "—"
                rows.append(
                    {
                        "Engine": label,
                        "Time": t_s,
                        "Velocity": vf_s,
                        "Peak RAM": peak,
                        "Stress": stress,
                    }
                )
            heading = (
                "Single Month · Uncapped"
                if mode == "small"
                else f"Full Window · {cap} MB Cap"
            )
            md = (
                f"#### {title} — {heading}\n\n"
                + pd.DataFrame(rows).to_markdown(index=False)
                + f"\n\n*{self._verify.get(mode, '')}*"
            )
            display(Markdown(md))

    # ------------------------------------------------------------------
    def export_results(
        self,
        suite_id,
        title,
        core_pattern,
        description,
        output_path="docs/benchmark_results.json",
    ):
        path = os.path.join(dataio.project_root(), output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)

        runs: Dict[str, Any] = {}
        for mode, res in self.results.items():
            engines = {}
            for label, r in res.items():
                e = {"status": r["status"], "peak_mb": round(r["peak_mb"], 1)}
                if r["status"] == "ok":
                    e["time"] = round(r["time"], 6)
                engines[label] = e
            runs[mode] = {
                "cap_mb": self.large_cap_mb if mode == "large" else None,
                "n_months": (
                    len(self.small_paths) if mode == "small" else len(self.large_paths)
                ),
                "iterations": self.large_iterations
                if mode == "large"
                else self.iterations,
                "warmup": self.large_warmup if mode == "large" else self.warmup,
                "engines": engines,
                "verify": self._verify.get(mode, ""),
            }

        existing[suite_id] = {
            "title": title,
            "core_pattern": core_pattern,
            "description": description,
            "small_iterations": self.iterations,
            "small_warmup": self.warmup,
            "large_iterations": self.large_iterations,
            "large_warmup": self.large_warmup,
            "runs": runs,
        }
        existing["environment"] = self._env_block(existing.get("environment"))
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"📊 exported suite [{suite_id}] → {output_path}")

    def _env_block(self, prior):
        mem = psutil.virtual_memory()
        env = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_model": _cpu_model(),
            "cpu_cores": os.cpu_count(),
            "total_ram_gb": round(
                mem.total / getattr(config, "BYTES_PER_GB", 1024**3), 2
            ),
            "python_version": platform.python_version(),
            "large_cap_mb": self.large_cap_mb,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        return env


def _cpu_model() -> str:
    try:
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return platform.machine()
