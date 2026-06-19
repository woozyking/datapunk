"""
Data acquisition and preparation: monthly file resolution, downloading,
parquet schema unification across months, and OS page-cache warming.
"""

from __future__ import annotations

import functools
import os
import urllib.request
from typing import Dict, List, Tuple

from . import config


@functools.lru_cache(maxsize=1)
def project_root() -> str:
    """
    Locate the repo root robustly, regardless of how deep this module is nested
    (datapunk/, src/datapunk/, an editable install, …).

    Order of resolution:
      1. ``$DATAPUNK_ROOT`` if set (handy for pointing data/ at a big disk).
      2. Walk up from this file until a ``pyproject.toml`` (or ``.git``) is found.
      3. Fall back to the current working directory.
    """
    env = os.environ.get("DATAPUNK_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    while True:
        if os.path.exists(os.path.join(cur, "pyproject.toml")) or os.path.isdir(
            os.path.join(cur, ".git")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            return os.getcwd()
        cur = parent


def data_dir() -> str:
    d = os.path.join(project_root(), "data")
    os.makedirs(d, exist_ok=True)
    return d


def monthly_files(
    start: Tuple[int, int] = config.START,
    end: Tuple[int, int] = config.END,
) -> List[Tuple[str, str]]:
    """Return chronological ``[(local_path, url), ...]`` for the month range."""
    (sy, sm), (ey, em) = start, end
    pairs: List[Tuple[str, str]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        url = config.TRIP_URL_TEMPLATE.format(year=y, month=m)
        local = os.path.join(data_dir(), url.rsplit("/", 1)[-1])
        pairs.append((local, url))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return pairs


def lookup_file() -> Tuple[str, str]:
    return os.path.join(data_dir(), config.LOOKUP_FILENAME), config.LOOKUP_URL


def bootstrap(pairs: List[Tuple[str, str]]) -> List[str]:
    """Download any missing files; return local paths."""
    paths = []
    for local, url in pairs:
        if not os.path.exists(local):
            print(f"  ⬇ downloading {os.path.basename(local)} …")
            urllib.request.urlretrieve(url, local)
            print(f"     ✅ {os.path.getsize(local) / 1e6:.1f} MB")
        paths.append(local)
    return paths


def warm_page_cache(paths: List[str], chunk: int = 1 << 22) -> None:
    """
    Read files once so subsequent reads hit the OS page cache. The cache is
    kernel-wide, so this primes reads for every isolated worker — done once in
    the parent rather than redundantly per engine.
    """
    for p in paths:
        try:
            with open(p, "rb") as f:
                while f.read(chunk):
                    pass
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Schema unification across months (some columns/types drift between files).
# ---------------------------------------------------------------------------


def unify_parquet_schema(file_paths: List[str]) -> List[str]:
    """
    Rewrite parquet files to a unified schema (union of columns, common
    supertypes, all timestamps normalised to ns) so strict multi-file readers
    can consume them in one call. Rewritten files are cached with a
    ``.unified.parquet`` suffix and reused.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schemas = [pq.read_schema(p) for p in file_paths]
    unified_fields: Dict[str, "pa.Field"] = {}
    for s in schemas:
        for f in s:
            fn = _normalise_field(f)
            if fn.name not in unified_fields:
                unified_fields[fn.name] = fn
            elif unified_fields[fn.name].type != fn.type:
                resolved = _common_supertype(unified_fields[fn.name].type, fn.type)
                unified_fields[fn.name] = pa.field(fn.name, resolved)
    unified = pa.schema(list(unified_fields.values()))

    if all(s == unified for s in schemas):
        return file_paths

    print(
        f"  ⚙ unifying schema across {len(file_paths)} files "
        f"({len(unified.names)} cols)…"
    )
    out: List[str] = []
    for p in file_paths:
        up = p.replace(".parquet", ".unified.parquet")
        # Rewrite if cached file is stale (schema doesn't match current unified).
        if os.path.exists(up):
            try:
                if pq.read_schema(up) == unified:
                    out.append(up)
                    continue
            except Exception:
                pass
        table = pq.read_table(p)
        arrays = []
        for name in unified.names:
            target = unified.field(name).type
            if name in table.column_names:
                col = table.column(name)
                if table.schema.field(name).type != target:
                    try:
                        col = pa.compute.cast(col, target, safe=True)
                    except Exception:
                        col = pa.compute.cast(col, target, safe=False)
                arrays.append(col)
            else:
                arrays.append(pa.nulls(table.num_rows, type=target))
        pq.write_table(pa.table(dict(zip(unified.names, arrays)), schema=unified), up)
        out.append(up)
    print("     ✅ unified files cached.")
    return out


def _normalise_field(f):
    import pyarrow as pa

    if pa.types.is_timestamp(f.type):
        return pa.field(f.name, pa.timestamp("ns"))
    return f


def _common_supertype(a, b):
    import pyarrow as pa

    if a == b:
        return a
    if pa.types.is_timestamp(a) and pa.types.is_timestamp(b):
        return pa.timestamp("ns")
    if pa.types.is_integer(a) and pa.types.is_integer(b):
        return pa.int64()
    if pa.types.is_floating(a) and pa.types.is_floating(b):
        return pa.float64()
    if (pa.types.is_integer(a) and pa.types.is_floating(b)) or (
        pa.types.is_floating(a) and pa.types.is_integer(b)
    ):
        return pa.float64()
    return pa.string()
