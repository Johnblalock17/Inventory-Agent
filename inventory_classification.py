"""
Multicriteria inventory classification (Sales and Hits, weighted score) with pandas.

Method
------
1. For Sales ($) and for Hits separately: sort items from highest to lowest, compute each item's
   cumulative % of the total, and assign a class:
       cumulative % <= 60  -> A
       cumulative % <= 80  -> B
       cumulative % <= 90  -> C
       cumulative % >  90  -> D
2. Convert each class to a score: A = 4, B = 3, C = 2, D = 1.
3. Combined Weighted Score = 0.65 x Sales Score + 0.35 x Hits Score.
4. Final class:  score >= 3.25 -> A;  2.50-3.24 -> B;  1.75-2.49 -> C;  below 1.75 -> D.

Every number is computed from the uploaded dataframe; nothing is hard-coded from a sample file.
"""

import re
from dataclasses import dataclass, field
from typing import List

import pandas as pd

REQUIRED_COLUMNS = ["Item Number", "Item Name", "Sales ($)", "Hits"]
OPTIONAL_COLUMNS = ["UoM"]
CLASSES = ["A", "B", "C", "D"]

# Step 1: (upper limit of cumulative %, class), checked in order
CUMULATIVE_LIMITS = [(60.0, "A"), (80.0, "B"), (90.0, "C")]      # above 90% -> D
# Step 2
CLASS_SCORE = {"A": 4, "B": 3, "C": 2, "D": 1}
# Step 3
SALES_WEIGHT = 0.65
HITS_WEIGHT = 0.35
# Step 4: (minimum combined score, final class), checked in order
FINAL_THRESHOLDS = [(3.25, "A"), (2.50, "B"), (1.75, "C")]        # below 1.75 -> D


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


def class_from_cumulative(cum_pct: float) -> str:
    """Step 1: cumulative % -> A/B/C/D (limits inclusive)."""
    cum_pct = round(float(cum_pct), 9)                  # 60.000000001 from float sums counts as 60
    for limit, cls in CUMULATIVE_LIMITS:
        if cum_pct <= limit:
            return cls
    return "D"


def combined_score(sales_score: float, hits_score: float) -> float:
    """Step 3, rounded to 2 decimals (the precision of the step-4 thresholds)."""
    return round(SALES_WEIGHT * sales_score + HITS_WEIGHT * hits_score, 2)


def final_class(score: float) -> str:
    """Step 4."""
    for minimum, cls in FINAL_THRESHOLDS:
        if score >= minimum:
            return cls
    return "D"


def single_criterion(values: pd.Series) -> pd.DataFrame:
    """Step 1 for one criterion: sort high to low, cumulative % of total, class.  Aligned to values.index."""
    order = values.sort_values(ascending=False, kind="mergesort")           # stable: ties keep file order
    positive = order.clip(lower=0)
    total = positive.sum()
    cum = positive.cumsum() / total * 100 if total > 0 else pd.Series(100.0, index=order.index)
    out = pd.DataFrame({"cum": cum, "class": cum.map(class_from_cumulative)}, index=order.index)
    return out.reindex(values.index)


@dataclass
class ClassificationResult:
    data: pd.DataFrame              # one row per item with both classes, both scores and the final class
    summary: pd.DataFrame           # Class, Items, Total Sales ($) for A-D
    skipped_rows: List[int] = field(default_factory=list)
    negative_rows: List[int] = field(default_factory=list)


def classify(df: pd.DataFrame) -> ClassificationResult:
    data = standardize(df)

    bad = data[["Sales ($)", "Hits"]].isna().any(axis=1)
    skipped = [int(i) + 2 for i in data.index[bad]]                         # CSV line numbers (header = 1)
    negative = [int(i) + 2 for i in data.index[(~bad) & ((data["Sales ($)"] < 0) | (data["Hits"] < 0))]]
    items = data[~bad].copy()
    if items.empty:
        raise ClassificationError("No rows have numeric values in both Sales ($) and Hits.")

    sales = single_criterion(items["Sales ($)"])
    hits = single_criterion(items["Hits"])
    items["Sales Cumulative %"] = sales["cum"]
    items["Sales Class"] = sales["class"]
    items["Sales Score"] = items["Sales Class"].map(CLASS_SCORE)
    items["Hits Cumulative %"] = hits["cum"]
    items["Hits Class"] = hits["class"]
    items["Hits Score"] = items["Hits Class"].map(CLASS_SCORE)
    items["Combined Weighted Score"] = [combined_score(s, h) for s, h in zip(items["Sales Score"], items["Hits Score"])]
    items["Final Class"] = items["Combined Weighted Score"].map(final_class)

    items = items.sort_values(["Final Class", "Combined Weighted Score", "Sales ($)"],
                              ascending=[True, False, False], kind="mergesort").reset_index(drop=True)

    grouped = items.groupby("Final Class").agg(Items=("Final Class", "size"), Sales=("Sales ($)", "sum"))
    grouped = grouped.reindex(CLASSES, fill_value=0)
    summary = pd.DataFrame({"Class": CLASSES,
                            "Items": grouped["Items"].astype(int).values,
                            "Total Sales ($)": grouped["Sales"].values})

    return ClassificationResult(items, summary, skipped, negative)
