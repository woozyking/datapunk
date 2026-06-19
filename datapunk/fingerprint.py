"""
Result fingerprinting for cross-engine consistency checks.

The timed benchmark functions return each engine's native *materialized* output.
Fingerprinting happens after timing and reduces that native result to a small,
order-independent summary: row count plus per-column aggregate statistics.

Where practical, this module fingerprints in the result's native engine instead
of first exporting the whole result to Arrow/Pandas. That keeps verification from
turning into a second, hidden materialization benchmark and avoids punishing
engines that deliberately keep results in their own memory format.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pyarrow as pa
import pyarrow.compute as pc


@dataclass
class DuckDBTableResult:
    """A native DuckDB materialized-result handle.

    Benchmark functions can return this after running `CREATE TEMP TABLE ... AS`
    so the timed output is materialized in DuckDB itself, not exported to Arrow
    or pandas. The open connection owns the session-scoped temp table.
    """

    con: Any
    table: str

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass


def close_result(df: Any) -> None:
    """Release native resources held only for post-run verification."""
    if isinstance(df, DuckDBTableResult):
        df.close()


def fingerprint(df: Any, target_cols: List[str]) -> Dict[str, Any]:
    """Reduce *df* (restricted to *target_cols*) to a comparable summary dict."""
    if isinstance(df, DuckDBTableResult):
        return _fingerprint_duckdb_table(df, target_cols)

    # Dask `.compute()` returns pandas; pandas also covers pandas baselines.
    if _is_pandas_dataframe(df):
        return _fingerprint_pandas(df, target_cols)

    # Polars `.collect()` returns an eager Polars DataFrame.
    if _is_polars_dataframe(df):
        return _fingerprint_polars(df, target_cols)

    # Fallback for Arrow, Daft, and any future engine that exposes to_arrow().
    return _fingerprint_arrow(_to_arrow(df), target_cols)


def to_pandas_frame(df: Any, target_cols: Sequence[str] | None = None):
    """Convert a native result to pandas for notebook visualization only."""
    import pandas as pd

    if isinstance(df, tuple) and len(df) == 2 and isinstance(df[1], dict):
        df = df[0]

    if isinstance(df, DuckDBTableResult):
        cols = _duckdb_target_cols(df, list(target_cols) if target_cols else None)
        select_sql = ", ".join(_quote_ident(c) for c in cols) if cols else "*"
        return df.con.execute(
            f"SELECT {select_sql} FROM {_quote_ident(df.table)}"
        ).fetchdf()

    if isinstance(df, pa.Table):
        out = df.to_pandas()
    elif hasattr(df, "to_pandas"):
        out = df.to_pandas()
    elif hasattr(df, "to_arrow"):
        arrow = df.to_arrow()
        if not isinstance(arrow, pa.Table):
            arrow = pa.table(arrow)
        out = arrow.to_pandas()
    else:
        out = pd.DataFrame(df)

    if target_cols is not None:
        cols = [c for c in target_cols if c in out.columns]
        out = out[cols]
    return out


# ---------------------------------------------------------------------------
# Native fingerprint adapters
# ---------------------------------------------------------------------------


def _fingerprint_duckdb_table(
    res: DuckDBTableResult, target_cols: List[str]
) -> Dict[str, Any]:
    con = res.con
    table_sql = _quote_ident(res.table)
    schema = _duckdb_schema(res)
    cols = [c for c in target_cols if c in schema]
    n_rows = int(con.execute(f"SELECT COUNT(*) FROM {table_sql}").fetchone()[0])
    fp: Dict[str, Any] = {"n_rows": n_rows, "cols": {}}

    for name in cols:
        col_sql = _quote_ident(name)
        dtype = schema[name].upper()
        stat_expr = _duckdb_stat_expr(col_sql, dtype)
        if _duckdb_is_numeric_like(dtype):
            total, min_v, max_v, mean_v, nulls = con.execute(
                f"""
                SELECT
                    SUM({stat_expr}) AS sum_v,
                    MIN({stat_expr}) AS min_v,
                    MAX({stat_expr}) AS max_v,
                    AVG({stat_expr}) AS mean_v,
                    COUNT(*) - COUNT({col_sql}) AS nulls
                FROM {table_sql}
                """
            ).fetchone()
            fp["cols"][name] = {
                "kind": "num",
                "sum": 0.0 if total is None else float(total),
                "min": _as_optional_float(min_v),
                "max": _as_optional_float(max_v),
                "mean": _as_optional_float(mean_v),
                "nulls": int(nulls),
            }
        else:
            rows = con.execute(
                f"""
                SELECT CAST({col_sql} AS VARCHAR) AS value, COUNT(*) AS count
                FROM {table_sql}
                GROUP BY 1
                ORDER BY 1
                """
            ).fetchall()
            pairs = sorted((str(v), int(c)) for v, c in rows)
            h = hashlib.sha1(repr(pairs).encode()).hexdigest()[:16]
            nulls = con.execute(
                f"SELECT COUNT(*) - COUNT({col_sql}) FROM {table_sql}"
            ).fetchone()[0]
            fp["cols"][name] = {"kind": "cat", "hash": h, "nulls": int(nulls)}
    return fp


def _fingerprint_pandas(df: Any, target_cols: List[str]) -> Dict[str, Any]:
    import pandas as pd
    from pandas.api import types as pdt

    cols = [c for c in target_cols if c in df.columns]
    fp: Dict[str, Any] = {"n_rows": int(len(df)), "cols": {}}
    for name in cols:
        s = df[name]
        nulls = int(s.isna().sum())
        # Booleans are numeric for summary purposes. Temporal columns are
        # treated as categorical distributions so equivalent Date/Timestamp
        # outputs from different engines do not disagree over storage units.
        if pdt.is_bool_dtype(s):
            s2 = s.astype("Int8")
        else:
            s2 = s

        if pdt.is_numeric_dtype(s2):
            fp["cols"][name] = {
                "kind": "num",
                "sum": _pandas_sum_as_float(s2),
                "min": _as_optional_float(s2.min(skipna=True)),
                "max": _as_optional_float(s2.max(skipna=True)),
                "mean": _as_optional_float(s2.mean(skipna=True)),
                "nulls": nulls,
            }
        else:
            vc = s.astype("string").value_counts(dropna=False)
            pairs = sorted((str(k), int(v)) for k, v in vc.items())
            h = hashlib.sha1(repr(pairs).encode()).hexdigest()[:16]
            fp["cols"][name] = {"kind": "cat", "hash": h, "nulls": nulls}
    return fp


def _fingerprint_polars(df: Any, target_cols: List[str]) -> Dict[str, Any]:
    import polars as pl

    cols = [c for c in target_cols if c in df.columns]
    fp: Dict[str, Any] = {"n_rows": int(df.height), "cols": {}}
    for name in cols:
        s = df.get_column(name)
        dtype_s = str(s.dtype)
        # Booleans are numeric for summary purposes. Temporal columns are
        # left as values and hashed categorically to avoid unit mismatches
        # across Date/Timestamp representations.
        if dtype_s == "Boolean":
            s2 = s.cast(pl.Int8)
        else:
            s2 = s

        if _polars_is_numeric(str(s2.dtype)):
            fp["cols"][name] = {
                "kind": "num",
                "sum": _polars_sum_as_float(s2),
                "min": _as_optional_float(s2.min()),
                "max": _as_optional_float(s2.max()),
                "mean": _as_optional_float(s2.mean()),
                "nulls": int(s.null_count()),
            }
        else:
            vc = s.value_counts().to_dicts()
            pairs = sorted((str(row.get(name)), int(row["count"])) for row in vc)
            h = hashlib.sha1(repr(pairs).encode()).hexdigest()[:16]
            fp["cols"][name] = {"kind": "cat", "hash": h, "nulls": int(s.null_count())}
    return fp


# ---------------------------------------------------------------------------
# Arrow fallback and comparison
# ---------------------------------------------------------------------------


def _to_arrow(df: Any) -> pa.Table:
    """Coerce a result into a PyArrow table as a fallback verification path."""
    if isinstance(df, pa.Table):
        return df
    if hasattr(df, "to_arrow"):
        out = df.to_arrow()
        return out if isinstance(out, pa.Table) else pa.table(out)
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    return pa.Table.from_pandas(df, preserve_index=False)


def _fingerprint_arrow(table: pa.Table, target_cols: List[str]) -> Dict[str, Any]:
    cols = [c for c in target_cols if c in table.column_names]
    table = table.select(cols)

    fp: Dict[str, Any] = {"n_rows": table.num_rows, "cols": {}}
    for name in cols:
        col = table.column(name)
        t = col.type
        # Temporal Arrow columns are hashed categorically rather than cast to
        # integer storage units, which differ between Date and Timestamp types.
        if pa.types.is_boolean(t):
            col = pc.cast(col, pa.int8())
            t = col.type

        if pa.types.is_integer(t) or pa.types.is_floating(t):
            fp["cols"][name] = {
                "kind": "num",
                "sum": float(pc.sum(col).as_py() or 0.0),
                "min": _as_float(pc.min(col)),
                "max": _as_float(pc.max(col)),
                "mean": _as_float(pc.mean(col)),
                "nulls": col.null_count,
            }
        else:
            vc = pc.value_counts(col)
            pairs = sorted((str(s["values"]), s["counts"]) for s in vc.to_pylist())
            h = hashlib.sha1(repr(pairs).encode()).hexdigest()[:16]
            fp["cols"][name] = {"kind": "cat", "hash": h, "nulls": col.null_count}
    return fp


def compare(a: Dict[str, Any], b: Dict[str, Any], rel_tol=1e-6, abs_tol=1e-6):
    """Return (ok, reason). Compares two fingerprints with float tolerance."""
    if a["n_rows"] != b["n_rows"]:
        return False, f"row count {a['n_rows']} != {b['n_rows']}"
    if set(a["cols"]) != set(b["cols"]):
        return False, f"columns {set(a['cols'])} != {set(b['cols'])}"
    for name, ca in a["cols"].items():
        cb = b["cols"][name]
        if ca["kind"] != cb["kind"]:
            return False, f"{name}: kind {ca['kind']} != {cb['kind']}"
        if ca.get("nulls") != cb.get("nulls"):
            return False, f"{name}.nulls: {ca.get('nulls')} != {cb.get('nulls')}"
        if ca["kind"] == "cat":
            if ca["hash"] != cb["hash"]:
                return False, f"{name}: categorical distribution differs"
        else:
            for stat in ("sum", "min", "max", "mean"):
                x, y = ca[stat], cb[stat]
                if x is None and y is None:
                    continue
                if x is None or y is None or not _isclose(x, y, rel_tol, abs_tol):
                    return False, f"{name}.{stat}: {x} != {y}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _duckdb_schema(res: DuckDBTableResult) -> Dict[str, str]:
    rows = res.con.execute(
        f"DESCRIBE SELECT * FROM {_quote_ident(res.table)}"
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _duckdb_target_cols(
    res: DuckDBTableResult, target_cols: List[str] | None
) -> List[str]:
    schema_cols = list(_duckdb_schema(res))
    if target_cols is None:
        return schema_cols
    return [c for c in target_cols if c in schema_cols]


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _duckdb_is_numeric_like(dtype: str) -> bool:
    dtype = dtype.upper()
    numeric_tokens = (
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
        "FLOAT",
        "DOUBLE",
        "REAL",
        "DECIMAL",
    )
    return dtype == "BOOLEAN" or any(tok in dtype for tok in numeric_tokens)


def _duckdb_is_temporal(dtype: str) -> bool:
    dtype = dtype.upper()
    return (
        dtype.startswith("DATE")
        or dtype.startswith("TIMESTAMP")
        or dtype.startswith("TIME")
    )


def _duckdb_stat_expr(col_sql: str, dtype: str) -> str:
    dtype = dtype.upper()
    if dtype == "BOOLEAN":
        return f"CAST({col_sql} AS TINYINT)"
    if dtype.startswith("TIMESTAMP"):
        return f"epoch_ns({col_sql})"
    if dtype.startswith("DATE"):
        return f"epoch({col_sql})"
    if dtype.startswith("TIME"):
        return f"epoch_ns({col_sql})"
    return col_sql


def _is_pandas_dataframe(df: Any) -> bool:
    return type(df).__module__.startswith("pandas.") and hasattr(df, "columns")


def _is_polars_dataframe(df: Any) -> bool:
    return type(df).__module__.startswith("polars.") and hasattr(df, "get_column")


def _polars_is_numeric(dtype_s: str) -> bool:
    return dtype_s.startswith(("Int", "UInt", "Float", "Decimal"))


def _polars_is_temporal(dtype_s: str) -> bool:
    return dtype_s.startswith(("Date", "Datetime", "Time", "Duration"))


def _pandas_sum_as_float(s: Any) -> float:
    v = s.sum(skipna=True)
    return 0.0 if v is None else float(v)


def _polars_sum_as_float(s: Any) -> float:
    v = s.sum()
    return 0.0 if v is None else float(v)


def _as_float(scalar) -> float | None:
    v = scalar.as_py()
    return None if v is None else float(v)


def _as_optional_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        import pandas as pd

        if pd.isna(v):
            return None
    except Exception:
        try:
            if v != v:  # NaN fallback without a pandas dependency.
                return None
        except Exception:
            pass
    return float(v)


def _isclose(x, y, rel_tol, abs_tol):
    return abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
