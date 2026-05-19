from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CSV_COLUMNS
from .utils import ensure_dir


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {}
        for col in CSV_COLUMNS:
            value = record.get(col, "")
            if value is None:
                value = ""
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def write_csv(records: list[dict], output_csv_path: str | Path) -> Path:
    output_csv_path = Path(output_csv_path)
    ensure_dir(output_csv_path.parent)
    df = records_to_dataframe(records)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    return output_csv_path
