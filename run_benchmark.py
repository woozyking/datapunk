"""
Datapunk Benchmark CLI.

    uv run python run_benchmark.py  # run all notebooks, write output JSON used by dashboard (docs/index.html)
"""

import glob
import json
import os
import subprocess
import sys

NOTEBOOK_DIR = "notebooks"
DOCS_DIR = "docs"
RESULTS_PATH = os.path.join(DOCS_DIR, "benchmark_results.json")


def execute_notebooks():
    notebooks = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "*.ipynb")))
    if not notebooks:
        print(f"No notebooks in {NOTEBOOK_DIR}/.")
        sys.exit(1)
    os.makedirs(DOCS_DIR, exist_ok=True)
    for nb in notebooks:
        print(f"Executing {nb} …")
        cmd = [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=1800",
            nb,
        ]
        env = {**os.environ, "DATAPUNK_NOTEBOOK_PATH": os.path.abspath(nb)}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"❌ {nb}\n{r.stderr}")
            sys.exit(1)
        print(f"  ✅ {nb}")


def validate_json():
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ no results at {RESULTS_PATH}.")
        sys.exit(1)
    data = json.load(open(RESULTS_PATH))
    suites = [k for k in data if k != "environment"]
    print(
        f"✅ {RESULTS_PATH} valid — {len(suites)} suite(s): {', '.join(sorted(suites))}"
    )


def main():
    print("=" * 56)
    print("  Datapunk Benchmark CLI")
    print("=" * 56)
    execute_notebooks()
    validate_json()
    print("Done.")


if __name__ == "__main__":
    main()
