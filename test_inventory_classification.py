"""Multicriteria classification (Sales and Hits, weighted score): every rule of the required method."""

import io
import itertools
import os
import sys

import pytest

pd = pytest.importorskip("pandas")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import csv_analysis  # noqa: E402
import inventory_classification as ic  # noqa: E402

# Hand-worked example.
# Sales, sorted high->low (total 1000): P1 600 -> 60% A | P2 200 -> 80% B | P3 100 -> 90% C | P4 60 -> 96% D | P5 40 -> 100% D
# Hits,  sorted high->low (total 100):  P5 40  -> 40% A | P4 30  -> 70% B | P3 15  -> 85% C | P1 10 -> 95% D | P2 5  -> 100% D
# Combined = 0.65 x Sales Score + 0.35 x Hits Score:
#   P1 S=A(4) H=D(1): 2.60 + 0.35 = 2.95 -> B
#   P2 S=B(3) H=D(1): 1.95 + 0.35 = 2.30 -> C
#   P3 S=C(2) H=C(2): 1.30 + 0.70 = 2.00 -> C
#   P4 S=D(1) H=B(3): 0.65 + 1.05 = 1.70 -> D
#   P5 S=D(1) H=A(4): 0.65 + 1.40 = 2.05 -> C
CSV = """Item Number,Item Name,UoM,Sales ($),Hits
P1,One,EA,600,10
P2,Two,EA,200,5
P3,Three,EA,100,15
P4,Four,EA,60,30
P5,Five,EA,40,40
"""
EXPECTED = {  # item: (sales class, hits class, sales score, hits score, combined, final)
    "P1": ("A", "D", 4, 1, 2.95, "B"),
    "P2": ("B", "D", 3, 1, 2.30, "C"),
    "P3": ("C", "C", 2, 2, 2.00, "C"),
    "P4": ("D", "B", 1, 3, 1.70, "D"),
    "P5": ("D", "A", 1, 4, 2.05, "C"),
}


def _classify(text=CSV):
    return ic.classify(csv_analysis.read_csv(io.StringIO(text)))


# ---- step 1: cumulative % classes ------------------------------------------------------------

@pytest.mark.parametrize("cum, cls", [(0.1, "A"), (60, "A"), (60.01, "B"), (80, "B"), (80.01, "C"),
                                      (90, "C"), (90.01, "D"), (100, "D")])
def test_cumulative_limits_are_inclusive(cum, cls):
    assert ic.class_from_cumulative(cum) == cls


def test_cumulative_percent_is_computed_on_sorted_values():
    d = _classify().data.set_index("Item Number")
    assert d["Sales Cumulative %"].to_dict() == pytest.approx({"P1": 60, "P2": 80, "P3": 90, "P4": 96, "P5": 100})
    assert d["Hits Cumulative %"].to_dict() == pytest.approx({"P5": 40, "P4": 70, "P3": 85, "P1": 95, "P2": 100})


def test_float_rounding_does_not_move_an_item_past_a_limit():
    # 0.1 + 0.2 + 0.3 style sums: three items of 20% each -> cumulative 60% exactly must still be A
    d = _classify("Item Number,Item Name,Sales ($),Hits\n"
                  "a,a,0.2,1\nb,b,0.2,1\nc,c,0.2,1\nd,d,0.2,1\ne,e,0.2,1\n").data.set_index("Item Number")
    assert d.loc["c", "Sales Class"] == "A"


# ---- steps 2-4: scores and final class --------------------------------------------------------

def test_class_scores():
    assert ic.CLASS_SCORE == {"A": 4, "B": 3, "C": 2, "D": 1}


@pytest.mark.parametrize("score, cls", [(4.0, "A"), (3.25, "A"), (3.24, "B"), (2.50, "B"), (2.49, "C"),
                                        (1.75, "C"), (1.74, "D"), (1.0, "D")])
def test_final_class_thresholds(score, cls):
    assert ic.final_class(score) == cls


def test_every_class_combination():
    for s, h in itertools.product("ABCD", repeat=2):
        expected = round(0.65 * ic.CLASS_SCORE[s] + 0.35 * ic.CLASS_SCORE[h], 2)
        assert ic.combined_score(ic.CLASS_SCORE[s], ic.CLASS_SCORE[h]) == expected
    assert ic.final_class(ic.combined_score(4, 2)) == "A"     # 3.30
    assert ic.final_class(ic.combined_score(4, 1)) == "B"     # 2.95
    assert ic.final_class(ic.combined_score(2, 3)) == "C"     # 2.35
    assert ic.final_class(ic.combined_score(1, 3)) == "D"     # 1.70


def test_hand_worked_example():
    d = _classify().data.set_index("Item Number")
    for item, (sc, hc, ss, hs, comb, final) in EXPECTED.items():
        row = d.loc[item]
        assert (row["Sales Class"], row["Hits Class"]) == (sc, hc), item
        assert (row["Sales Score"], row["Hits Score"]) == (ss, hs), item
        assert row["Combined Weighted Score"] == pytest.approx(comb), item
        assert row["Final Class"] == final, item


# ---- step 5: outputs ---------------------------------------------------------------------------

def test_summary_counts_and_sales_per_class():
    s = _classify().summary.set_index("Class")
    assert list(s.index) == ["A", "B", "C", "D"]
    assert s["Items"].to_dict() == {"A": 0, "B": 1, "C": 3, "D": 1}
    assert s["Total Sales ($)"].to_dict() == pytest.approx({"A": 0, "B": 600, "C": 340, "D": 60})


def test_item_table_has_required_columns():
    cols = set(_classify().data.columns)
    assert {"Sales Class", "Hits Class", "Sales Score", "Hits Score", "Combined Weighted Score",
            "Final Class"} <= cols


def test_old_method_is_gone():
    assert not hasattr(ic, "COMBINED_MATRIX")
    import inspect
    with pytest.raises(TypeError):
        ic.classify(csv_analysis.read_csv(io.StringIO(CSV)), 0.8, 0.95)   # no cutoff arguments any more
    assert "0.95" not in inspect.getsource(ic)


# ---- input handling ----------------------------------------------------------------------------

def test_headers_and_currency_text_are_tolerated():
    d = _classify(' item number ,ITEM NAME,uom,Sales,hits\nX,Ex,EA,"$1,000",4\nY,Why,EA,$0,0\n').data
    d = d.set_index("Item Number")
    assert d.loc["X", "Sales ($)"] == 1000 and d.loc["Y", "Sales Class"] == "D"


def test_missing_columns_named():
    df = csv_analysis.read_csv(io.StringIO("Item Number,Sales ($)\n1,5\n"))
    assert ic.missing_columns(df) == ["Item Name", "Hits"]
    with pytest.raises(ic.ClassificationError):
        ic.classify(df)


def test_non_numeric_rows_skipped():
    r = _classify(CSV + "P6,Six,EA,abc,3\n")
    assert r.skipped_rows == [7] and len(r.data) == 5


def test_sample_file_classifies_every_item():
    df = csv_analysis.read_csv(os.path.join(ROOT, "sample_classification.csv"))
    r = ic.classify(df)
    assert len(r.data) == len(df) == r.summary["Items"].sum()
    assert r.summary["Total Sales ($)"].sum() == pytest.approx(r.data["Sales ($)"].sum())


# ---- Streamlit page (uses the existing uploader's show_data_analysis) ---------------------------

def _page(csv_bytes):
    return f'''
import io, os, sys
sys.path.insert(0, os.getcwd())
import streamlit_app
buf = io.BytesIO({csv_bytes!r})
buf.name = "upload.csv"
streamlit_app.show_data_analysis(buf)
'''


def test_streamlit_shows_classification(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(ROOT)
    at = st_testing.AppTest.from_string(_page(CSV.encode()), default_timeout=30).run()
    assert not at.exception, at.exception
    assert "Inventory Classification (Sales and Hits)" in [s.value for s in at.subheader]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {**metrics, "Class A": "0 items", "Class B": "1 item", "Class C": "3 items", "Class D": "1 item"}
    assert not at.number_input, "the old cutoff inputs should be gone"
    assert not at.error


def test_streamlit_existing_analysis_still_works(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(ROOT)
    at = st_testing.AppTest.from_string(_page(b"Product,Category,Units,Unit Cost\nA,X,10,2\nB,Y,30,1\n"),
                                        default_timeout=30).run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total Units"] == "40" and metrics["Total Inventory Value"] == "$50.00"
    assert "Inventory Classification (Sales and Hits)" not in [s.value for s in at.subheader]


def test_streamlit_wrong_file_names_both_column_sets(monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(ROOT)
    at = st_testing.AppTest.from_string(_page(b"Foo,Bar\n1,2\n"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert "Hits" in at.error[0].value and "Unit Cost" in at.error[0].value
