"""
Multicriteria (Sales × Hits) ABC inventory classification with pandas.

Method
------
1. Rank items by Sales ($), highest first, and compute each item's cumulative share of total
   sales.  An item is class A while the cumulative share *before* it is below the A cutoff
   (default 80%), B while it is below the B cutoff (default 95%), otherwise C.  So the item
   that crosses a cutoff joins the higher class.
2. Do the same for Hits (number of order lines / picks).
3. Combine the two single-criterion classes with a matrix into A, B, C, D:

        Sales \\ Hits   A   B   C
              A         A   A   B
              B         A   B   C
              C         B   C   D

Every number is computed from the uploaded dataframe; nothing is hard-coded from a sample file.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

REQUIRED_COLUMNS = ["Item Number", "Item Name", "Sales ($)", "Hits"]
OPTIONAL_COLUMNS = ["UoM"]
CLASSES = ["A", "B", "C", "D"]

COMBINED_MATRIX: Dict[str, Dict[str, str]] = {      # COMBINED_MATRIX[sales_class][hits_class]
    "A": {"A": "A", "B": "A", "C": "B"},
    "B": {"A": "A", "B": "B", "C": "C"},
    "C": {"A": "B", "B": "C", "C": "D"},
}


class ClassificationError(ValueError):
    def __init__(self, message: str, missing: List[str] = None):
        super().__init__(message)
        self.missing = missing or []


def _norm(name: str) -> str:
    """'Sales ($)', 'sales', ' SALES $ ' -> 'sales'.  Lets headers vary in case, spacing and symbols."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def missing_columns(df: pd.DataFrame) -> List[str]:
    present = {_norm(c) for c in df.columns}
    return [c for c in REQUIRED_COLUMNS if _norm(c) not in present]


def has_classification_columns(df: pd.DataFrame) -> bool:
    return not missing_columns(df)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical column names; Sales ($) and Hits converted to numbers ($ and commas removed)."""
    missing = missing_columns(df)
    if missing:
        raise ClassificationError("Missing required column(s): " + ", ".join(missing) + ".", missing)
    canonical = {_norm(c): c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}
    out = df.rename(columns={c: canonical[_norm(c)] for c in df.columns if _norm(c) in canonical}).copy()
    for col in ("Sales ($)", "Hits"):
        cleaned = out[col].astype(str).str.replace(r"[$,\s]", "", regex=True)
        cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)      # (123) -> -123
        out[col] = pd.to_numeric(cleaned, errors="coerce")
    return out


def single_criterion_class(values: pd.Series, a_cutoff: float, b_cutoff: float) -> pd.DataFrame:
    """ABC on one criterion.  Returns rank, share %, cumulative %, and class, aligned to values.index."""
    order = values.sort_values(ascending=False, kind="mergesort")          # stable: ties keep file order
    total = order.clip(lower=0).sum()
    share = order.clip(lower=0) / total * 100 if total > 0 else order * 0.0
    cum = share.cumsum()
    before = cum - share                                                   # cumulative % before this item
    cls = pd.Series("C", index=order.index)
    cls[before < b_cutoff * 100] = "B"
    cls[before < a_cutoff * 100] = "A"
    cls[order <= 0] = "C"                                                  # no sales / no hits is never A or B
    out = pd.DataFrame({"rank": range(1, len(order) + 1), "share": share, "cum": cum, "class": cls},
                       index=order.index)
    return out.reindex(values.index)


@dataclass
class ClassificationResult:
    data: pd.DataFrame              # uploaded data + classification columns, sorted by class then sales
    summary: pd.DataFrame           # one row per class A-D
    matrix_counts: pd.DataFrame     # Sales class × Hits class item counts
    a_cutoff: float
    b_cutoff: float
    skipped_rows: List[int] = field(default_factory=list)
    negative_rows: List[int] = field(default_factory=list)


def classify(df: pd.DataFrame, a_cutoff: float = 0.80, b_cutoff: float = 0.95) -> ClassificationResult:
    if not 0 < a_cutoff < b_cutoff < 1:
        raise ClassificationError("Cutoffs must satisfy 0 < A cutoff < B cutoff < 100%.")
    data = standardize(df)

    bad = data[["Sales ($)", "Hits"]].isna().any(axis=1)
    skipped = [int(i) + 2 for i in data.index[bad]]                       # CSV line numbers (header = line 1)
    negative = [int(i) + 2 for i in data.index[(~bad) & ((data["Sales ($)"] < 0) | (data["Hits"] < 0))]]
    valid = data[~bad].copy()
    if valid.empty:
        raise ClassificationError("No rows have numeric values in both Sales ($) and Hits.")

    s = single_criterion_class(valid["Sales ($)"], a_cutoff, b_cutoff)
    h = single_criterion_class(valid["Hits"], a_cutoff, b_cutoff)
    valid["Sales Rank"] = s["rank"]
    valid["Sales Share %"] = s["share"]
    valid["Sales Cum %"] = s["cum"]
    valid["Sales Class"] = s["class"]
    valid["Hits Rank"] = h["rank"]
    valid["Hits Share %"] = h["share"]
    valid["Hits Cum %"] = h["cum"]
    valid["Hits Class"] = h["class"]
    valid["Sales-Hits"] = valid["Sales Class"] + valid["Hits Class"]
    valid["Class"] = [COMBINED_MATRIX[a][b] for a, b in zip(valid["Sales Class"], valid["Hits Class"])]

    valid = valid.sort_values(["Class", "Sales ($)", "Hits"], ascending=[True, False, False], kind="mergesort")

    total_sales = valid["Sales ($)"].clip(lower=0).sum()
    total_hits = valid["Hits"].clip(lower=0).sum()
    grouped = valid.groupby("Class").agg(Items=("Class", "size"),
                                         Sales=("Sales ($)", lambda x: x.clip(lower=0).sum()),
                                         Hits=("Hits", lambda x: x.clip(lower=0).sum()))
    grouped = grouped.reindex(CLASSES, fill_value=0)
    summary = pd.DataFrame({
        "Class": CLASSES,
        "Items": grouped["Items"].astype(int).values,
        "% of Items": (grouped["Items"] / len(valid) * 100).values,
        "Sales ($)": grouped["Sales"].values,
        "% of Sales": (grouped["Sales"] / total_sales * 100 if total_sales else grouped["Sales"] * 0).values,
        "Hits": grouped["Hits"].values,
        "% of Hits": (grouped["Hits"] / total_hits * 100 if total_hits else grouped["Hits"] * 0).values,
    })

    matrix = (pd.crosstab(valid["Sales Class"], valid["Hits Class"])
              .reindex(index=["A", "B", "C"], columns=["A", "B", "C"], fill_value=0))
    matrix.index.name, matrix.columns.name = "Sales class", "Hits class"

    return ClassificationResult(valid, summary, matrix, a_cutoff, b_cutoff, skipped, negative)
