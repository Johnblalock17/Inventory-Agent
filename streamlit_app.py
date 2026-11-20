"""
Streamlit front end for the Inventory Decision Agent.

    streamlit run streamlit_app.py

All numbers come from the deterministic Python tools in inventory_tools.py; this file only
lays them out.  An Anthropic API key is optional: if ANTHROPIC_API_KEY is set in
st.secrets (or the environment), Claude helps read the question; otherwise the
built-in rule-based parser is used.  Either way, Python does every calculation.
"""

import os
import sys
from typing import Any, Dict, Optional

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find the agent modules beside this file

import csv_analysis                              # noqa: E402  pandas CSV upload analysis
from app import EXAMPLES, _parse_answer          # app.py's CLI only runs under `python app.py`
from inventory_agent import FOLLOW_UPS, INPUT_LABELS, AgentResponse, InventoryAgent
from inventory_tools import SECTION_TITLES

API_KEY_NAME = "ANTHROPIC_API_KEY"               # the name the anthropic SDK reads


def _api_key() -> Optional[str]:
    """Read the key from Streamlit secrets, then the environment. Never hard-coded."""
    try:
        key = st.secrets.get(API_KEY_NAME)
    except Exception:                            # no secrets.toml at all
        key = None
    return key or os.environ.get(API_KEY_NAME) or None


@st.cache_resource
def _client(key: str) -> Any:
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=key)


def make_agent(use_llm: bool) -> InventoryAgent:
    key = _api_key()
    client = _client(key) if (use_llm and key) else None
    return InventoryAgent(use_llm=client is not None, client=client)


# --------------------------------------------------------------------------- rendering

def show_worked_solution(resp: AgentResponse) -> None:
    ws: Dict[str, Any] = resp.result["worked_solution"]
    titles = dict(SECTION_TITLES)

    with st.expander(titles["inputs_extracted"], expanded=True):
        st.dataframe(
            [{"Symbol": i["symbol"], "Input": i["label"], "Value": i["display"],
              "Source": "given" if i["source"] == "given" else "not given (default)"}
             for i in ws["inputs_extracted"]],
            hide_index=True, width="stretch")
        if resp.evidence:
            st.caption("Read from your question:")
            for key, ev in resp.evidence.items():
                if isinstance(ev, dict):
                    label, snippet = ev.get("label", key.replace("_", " ")), ev.get("text", "")
                else:
                    label, snippet = key.replace("_", " "), ev
                st.markdown(f"- **{label}**: “{snippet}”")
        if resp.unused_numbers:
            st.warning("Numbers in the question that were not used: "
                       + ", ".join(f"{u:g}" for u in resp.unused_numbers)
                       + ". Check whether any of them matter.")

    with st.expander(titles["variable_definitions"]):
        st.dataframe([{"Symbol": v["symbol"], "Meaning": v["meaning"], "Unit": v.get("unit", "")}
                      for v in ws["variable_definitions"]], hide_index=True, width="stretch")

    with st.expander(titles["formula"], expanded=True):
        st.code("\n".join(f["formula"] for f in ws["formula"]), language=None)

    with st.expander(titles["substituted"], expanded=True):
        st.code("\n".join(s["line"] for s in ws["substituted"]), language=None)

    with st.expander(titles["intermediate"]):
        st.dataframe([{"Symbol": r["symbol"], "Meaning": r["meaning"], "Value": r["display"]}
                      for r in ws["intermediate"]], hide_index=True, width="stretch")

    with st.expander(titles["final_answer"], expanded=True):
        final = ws["final_answer"]
        st.success(final["text"])
        values = final["values"]
        for start in range(0, len(values), 3):
            cols = st.columns(3)
            for col, v in zip(cols, values[start:start + 3]):
                col.metric(v["label"], v["display"])

    with st.expander(titles["verification_checks"]):
        for c in ws["verification_checks"]:
            icon = "✅" if c["passed"] else "❌"
            st.markdown(f"{icon} **{c['name']}**: {c['detail']}")

    with st.expander(titles["assumptions"]):
        notes = list(ws["assumptions"]) + [a for a in resp.assumptions if a not in ws["assumptions"]]
        for a in notes or ["None."]:
            st.markdown(f"- {a}")


def show_missing_form(agent: InventoryAgent, resp: AgentResponse) -> None:
    st.info("Some information is missing. I won't guess it. Please fill it in:\n\n"
            + "\n".join(f"- {INPUT_LABELS.get(m, m)}" for m in resp.missing))
    with st.form("missing_inputs"):
        answers = {}
        for missing in resp.missing:
            for key, prompt, _default in FOLLOW_UPS.get(missing, []):
                if prompt:
                    answers[key] = st.text_input(prompt, key=f"fu_{key}")
        submitted = st.form_submit_button("Calculate")
    if submitted:
        overrides: Dict[str, Any] = dict(resp.raw_extraction)
        for missing in resp.missing:
            for key, prompt, default in FOLLOW_UPS.get(missing, []):
                if prompt is None:
                    overrides.setdefault(key, default)
        try:
            for key, text in answers.items():
                if text.strip():
                    overrides[key] = _parse_answer(key, text)
        except (ValueError, IndexError):
            st.error("Please enter numbers only (price breaks as 0:5, 1000:4.8).")
            return
        st.session_state.resp = agent.run(resp.question, overrides=overrides, model_key=resp.model_key)
        st.rerun()


def show_response(agent: InventoryAgent, resp: AgentResponse) -> None:
    if resp.status == "ok":
        verified = "✓ Verified" if resp.verified else "⚠ Check failed"
        st.subheader(f"{resp.model_name}  ·  {verified}")
        st.caption(f"Why this model: {resp.identification_reason}")
        st.markdown(f"**Answer:** {resp.result['answer']}")
        st.markdown("### Worked solution")
        show_worked_solution(resp)
    elif resp.status == "needs_input":
        st.subheader(resp.model_name or "Missing information")
        show_missing_form(agent, resp)
    elif resp.status == "unsupported":
        st.warning(resp.explanation)
    else:
        st.error(resp.explanation or "Something went wrong while calculating.")

    with st.expander("Pipeline trace"):
        for i, t in enumerate(resp.trace, 1):
            st.text(f"{i}. {t}")


# --------------------------------------------------------------------------- CSV data analysis

def _money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _units(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def show_data_analysis(uploaded: Any) -> None:
    """Read the uploaded CSV with pandas, then show the data and the analysis."""
    st.divider()
    st.header("Data Analysis")
    st.caption(f"File: {getattr(uploaded, 'name', 'uploaded CSV')}")

    try:
        df = csv_analysis.read_csv(uploaded)
    except csv_analysis.CSVDataError as exc:
        st.error(str(exc))
        return

    st.subheader("Uploaded data")
    st.dataframe(df, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    st.markdown("**Column names:** " + ", ".join(f"`{c}`" for c in df.columns))

    missing = csv_analysis.missing_columns(df)
    if missing:
        st.error("This file is missing the required column(s): **" + ", ".join(missing) + "**.\n\n"
                 "Required columns: " + ", ".join(csv_analysis.REQUIRED_COLUMNS) + ".")
        return

    try:
        result = csv_analysis.analyze(df)
    except csv_analysis.CSVDataError as exc:
        st.error(str(exc))
        return

    if result.skipped_rows:
        st.warning("Rows with a non-numeric Units or Unit Cost were left out of the totals "
                   f"(CSV line{'s' if len(result.skipped_rows) > 1 else ''} "
                   + ", ".join(map(str, result.skipped_rows)) + ").")

    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Units", _units(result.total_units))
    m2.metric("Average Unit Cost", _money(result.average_unit_cost))
    m3.metric("Product with Highest Units", ", ".join(result.top_products),
              help=f"{_units(result.top_units)} units")
    m4.metric("Total Inventory Value", _money(result.total_inventory_value))
    st.caption(f"Highest Units: {_units(result.top_units)}"
               + (" (tie)" if len(result.top_products) > 1 else ""))

    st.markdown("**Total Units by Category**")
    st.dataframe(result.units_by_category, hide_index=True, width="stretch",
                 column_config={"Total Units": st.column_config.NumberColumn(format="localized")})

    st.markdown("**Data with Inventory Value** (Inventory Value = Units × Unit Cost)")
    st.dataframe(result.data, hide_index=True, width="stretch",
                 column_config={"Unit Cost": st.column_config.NumberColumn(format="dollar"),
                                "Inventory Value": st.column_config.NumberColumn(format="dollar"),
                                "Units": st.column_config.NumberColumn(format="localized")})


# --------------------------------------------------------------------------- page

def main() -> None:
    st.set_page_config(page_title="Inventory Decision Agent", page_icon="📦", layout="wide")
    st.title("📦 Inventory Decision Agent")
    st.write("Ask an EOQ, quantity discount, safety stock / reorder point, fixed-period review, "
             "or newsvendor question. Python does every calculation and shows its work.")

    with st.sidebar:
        has_key = _api_key() is not None
        use_llm = st.toggle("Use Claude to read the question", value=has_key, disabled=not has_key,
                            help=f"Needs {API_KEY_NAME} in Streamlit secrets. Math is always done in Python.")
        if not has_key:
            st.caption(f"No {API_KEY_NAME} found, so the built-in parser is used.")
        st.markdown("**Sample questions**")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex[:70] + "…", key=f"ex{i}", width="stretch"):
                st.session_state.question = ex
                st.session_state.pop("resp", None)

        st.divider()
        st.subheader("Data Upload")
        uploaded = st.file_uploader("Upload CSV File", type=["csv"], key="csv_upload")

    agent = make_agent(use_llm)
    with st.form("ask"):
        question = st.text_area("Your question", key="question", height=120,
                                placeholder="Annual demand is 1,000 units, ordering cost is $10 ...")
        asked = st.form_submit_button("Solve", type="primary")
    if asked:
        if question.strip():
            with st.spinner("Calculating..."):
                st.session_state.resp = agent.run(question.strip())
        else:
            st.warning("Please type a question first.")

    if "resp" in st.session_state:
        show_response(agent, st.session_state.resp)

    if uploaded is not None:
        show_data_analysis(uploaded)


main()
