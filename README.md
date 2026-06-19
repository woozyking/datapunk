# Datapunk

Synthetic benchmarks like TPC-H offer idealized metrics. Just as PC gamers reach for *Cyberpunk 2077* to stress a rig, data engineers need a real-world suite to see how engines behave on practical analytics — and, crucially, on a **small box**.

Datapunk is that benchmark "game."

## Engines

- [Pandas](https://pandas.pydata.org/) - baseline, no memory configuration
- [Polars](https://pola.rs/) — default in-memory + streaming engine twin
- [DuckDB](https://duckdb.org/) — default + spill-to-disk twin (explicit `memory_limit`)
- [Dask](https://dask.org/) — threaded scheduler
- [Daft](https://daft.ai/) — streaming architecture, no memory configuration

Each engine runs with out-of-the-box defaults. Where the engine supports memory
configuration, a "twin" variant is included that receives an explicit memory budget
and spill directory — same query, with and without the engine knowing its constraint.

## The two-run model

Every suite runs each engine **twice**:

1. **Small, uncapped** — a single month (~3M rows). No memory limit. This is the apples-to-apples comparison of speed and correctness when everything fits in RAM.
2. **Large, capped** — the full 24-month window (30M+ rows) under a *physical* memory ceiling. This surfaces out-of-core behaviour: engines that stream and spill survive; engines that must materialise everything **OOM** — and the OOM is recorded as a first-class result, not a crash.

## How the memory cap works

Each engine runs in its **own isolated subprocess** (fresh interpreter, no shared allocator state, no cross-engine import pollution). The parent enforces the cap with a **physical-RSS watchdog**: it polls the worker's resident memory and `SIGKILL`s it the instant it crosses the limit, recording an OOM.

## Metrics

- **Velocity** — speedup vs. the slowest *successful* engine in that run (best ★).
- **Peak RAM** — peak resident memory (watchdog max ⨄ kernel high-water mark).
- **Stress** — peak RAM ÷ cap (large run only; ≥1.0 / OOM means it hit the ceiling).

Verification compares an order-independent **fingerprint** (row count + per-column aggregate stats) across engines with floating-point tolerance — robust to legitimate tie-break and last-ULP differences that a strict row-by-row check flaked on.

## Layout

```
datapunk/         # installable package
  config.py       # dataset source, default cap, run modes
  dataio.py       # download, schema unification, page-cache warming
  runner.py       # subprocess + RSS watchdog (the cap)
  worker.py       # isolated per-engine entry point
  fingerprint.py  # tolerant cross-engine consistency check
  reporter.py     # two-run orchestration, scorecard, JSON export
notebooks/        # one suite per notebook; analytics functions laid out inline
run_benchmark.py
```

## Quickstart

```bash
uv run python run_benchmark.py
```

Results are written to `public/benchmark_results.json`, used by the dashboard at https://io.github.com/woozyking/datapunk.
