"""Table IO helpers with optional Parquet acceleration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def parquet_available() -> bool:
    """Return True when pandas can use a Parquet engine."""

    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _candidate_paths(path_base: str | Path) -> tuple[Path, Path]:
    """Return CSV and Parquet paths for a base path."""

    path = Path(path_base)
    if path.suffix == ".csv":
        csv_path = path
        parquet_path = path.with_suffix(".parquet")
    elif path.suffix == ".parquet":
        parquet_path = path
        csv_path = path.with_suffix(".csv")
    else:
        csv_path = path.with_suffix(".csv")
        parquet_path = path.with_suffix(".parquet")
    return csv_path, parquet_path


def read_table(
    path_base: str | Path,
    prefer_parquet_if_available: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read a table from Parquet when available, otherwise CSV."""

    csv_path, parquet_path = _candidate_paths(path_base)
    if prefer_parquet_if_available and parquet_available() and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, **kwargs)
    if parquet_path.exists() and parquet_available():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"No readable table found for {path_base}")


def write_table(
    df: pd.DataFrame,
    path_base: str | Path,
    formats: Iterable[str] = ("csv", "parquet"),
    index: bool = False,
) -> dict[str, Path]:
    """Write a dataframe in requested formats, falling back gracefully."""

    csv_path, parquet_path = _candidate_paths(path_base)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    requested = {fmt.lower().lstrip(".") for fmt in formats}
    if "csv" in requested:
        df.to_csv(csv_path, index=index)
        written["csv"] = csv_path
    if "parquet" in requested and parquet_available():
        df.to_parquet(parquet_path, index=index)
        written["parquet"] = parquet_path
    return written

