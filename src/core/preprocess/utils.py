"""
Common utility functions.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def ensure_dir(path):
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_with_encoding(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file by trying several encodings."""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not read the file with any supported encoding: {path}")


def read_excel_all_sheets(path: str) -> Dict[str, pd.DataFrame]:
    """Read all Excel sheets and return {sheet_name: DataFrame}."""
    xls = pd.ExcelFile(path)
    result = {}
    for sheet in xls.sheet_names:
        result[sheet] = pd.read_excel(xls, sheet_name=sheet)
    return result


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing line breaks and normalizing whitespace."""
    df = df.copy()
    df.columns = [
        str(col).strip().replace("\n", "").replace("\r", "")
        for col in df.columns
    ]
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
    return df


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to float, handling percent signs, spaces, and similar characters."""
    s = series.copy()
    if s.dtype in (np.float64, np.int64, np.float32, np.int32):
        return s.astype(float)
    s = s.astype(str).str.strip()
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d\.\-\+eE]", "", regex=True)
    s = s.replace(["", ".", "-", "+", "nan", "NaN", "None", "NA"], np.nan)
    return pd.to_numeric(s, errors="coerce")


def save_report_to_excel(report_dict: Dict[str, pd.DataFrame], output_path: str):
    """Save multiple DataFrames to separate sheets in one Excel workbook."""
    ensure_dir(Path(output_path).parent)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in report_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
