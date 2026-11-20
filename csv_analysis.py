"""
CSV inventory-data analysis with pandas.

Every number is computed from the dataframe read from the user's uploaded file;
nothing here knows about any particular sample dataset.
"""

from dataclasses import dataclass, field
from typing import IO, List, Optional, Union

import pandas as pd

REQUIRED_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]
NUMERIC_COLUMNS = ["Units", "Unit Cost"]


class CSVDataError(ValueError):
    """The file could not be read, or it lacks what the analysis needs."""

    def __init__(self, message: str, missing: Optional[List[str]] = None):
        super().__init__(message)
        self.missing = missing or []


def read_csv(source: Union[str, IO]) -> pd.DataFrame:
    """Read an uploaded CSV into a dataframe, with column names trimmed of stray spaces."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise CSVDataError("The file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise CSVDataError(f"The file could not be read as CSV ({exc}).")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def missing_columns(df: pd.DataFrame) -> List[str]:
    """Required columns not present (matched ignoring case and surrounding spaces)."""
    present = {c.strip().lower() for c in df.columns}
    return [c for c in REQUIRED_COLUMNS if c.lower() not in present]


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename required columns to their canonical spelling and make Units / Unit Cost numeric.

    Raises CSVDataError naming any missing columns.
    """
    missing = missing_columns(df)
    if missing:
        raise CSVDataError("Missing required column(s): " + ", ".join(missing) + ".", missing)
    canonical = {c.lower(): c for c in REQUIRED_COLUMNS}
    out = df.rename(columns={c: canonical[c.lower()] for c in df.columns if c.lower() in canonical}).copy()
    for col in NUMERIC_COLUMNS:
        cleaned = out[col].astype(str).str.replace(r"[$,\s]", "", regex=True)
        out[col] = pd.to_numeric(cleaned, errors="coerce")
    return out


@dataclass
class InventoryAnalysis:
    data: pd.DataFrame                      # uploaded data plus the Inventory Value column
    total_units: float
    average_unit_cost: float
    top_products: List[str]                 # product(s) with the highest Units (ties kept)
    top_units: float
    total_inventory_value: float
    units_by_category: pd.DataFrame         # columns: Category, Total Units
    skipped_rows: List[int] = field(default_factory=list)   # rows whose Units/Unit Cost were not numbers


def analyze(df: pd.DataFrame) -> InventoryAnalysis:
    """All calculations, in pandas, from the uploaded dataframe."""
    data = standardize(df)
    bad = data[NUMERIC_COLUMNS].isna().any(axis=1)
    skipped = [int(i) + 2 for i in data.index[bad]]          # +2 = CSV line number (header is line 1)
    valid = data[~bad]
    if valid.empty:
        raise CSVDataError("No rows have numeric values in both Units and Unit Cost.")

    data["Inventory Value"] = data["Units"] * data["Unit Cost"]

    top_units = valid["Units"].max()
    top_products = valid.loc[valid["Units"] == top_units, "Product"].astype(str).tolist()

    by_category = (valid.groupby("Category", dropna=False)["Units"].sum()
                   .reset_index().rename(columns={"Units": "Total Units"})
                   .sort_values("Total Units", ascending=False, ignore_index=True))

    return InventoryAnalysis(
        data=data,
        total_units=float(valid["Units"].sum()),
        average_unit_cost=float(valid["Unit Cost"].mean()),
        top_products=top_products,
        top_units=float(top_units),
        total_inventory_value=float(data["Inventory Value"].sum()),
        units_by_category=by_category,
        skipped_rows=skipped,
    )
