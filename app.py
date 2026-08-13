import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Message Intelligence Pipeline", page_icon="📨", layout="wide")

DATA_DIR = Path(__file__).parent

@st.cache_data
def load_json(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)

classification = load_json("./outputs/classification_results.json")
extraction = load_json("./outputs/extraction_results.json")
sensitive = load_json("./outputs/sensitive_detection_results.json")
mandatory = load_json("./outputs/mandatory_ids_results.json")
summary = load_json("./outputs/summary.json")
emb_summary = load_json("embedding_validation_summary.json")
emb_disagreements = load_json("embedding_validation_disagreements.json")

class_df = pd.DataFrame(classification)
extract_df = pd.DataFrame(extraction)
sens_df = pd.DataFrame(sensitive)
mand_df = pd.DataFrame(mandatory)

# ---------------------------------------------------------------
st.title("📨 Message Intelligence Pipeline")
st.caption(
    "Rule-based classification, task/event extraction, and sensitive-info detection "
    "over 900 fictional messages — with a TF-IDF embedding layer as an independent cross-check."
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total messages", summary["total_messages"])
col2.metric("Tasks extracted", summary["total_tasks_extracted"])
col3.metric("Events extracted", summary["total_events_extracted"])
col4.metric("Sensitive flagged", summary["total_sensitive_flagged"])
col5.metric("Mandatory IDs shown", f"{summary['mandatory_ids_covered']}/15")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Classification", "Tasks & Events", "Sensitive Detection", "Embedding Validation", "Mandatory IDs"
])

# ---------------------------------------------------------------
with tab1:
    st.subheader("Category distribution")
    cat_counts = class_df["category"].value_counts()
    st.bar_chart(cat_counts)

    st.subheader("Browse classifications")
    categories = ["All"] + sorted(class_df["category"].unique().tolist())
    picked = st.selectbox("Filter by category", categories)
    view = class_df if picked == "All" else class_df[class_df["category"] == picked]
    st.dataframe(view, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
with tab2:
    st.subheader("Extracted tasks and events")
    type_filter = st.radio("Type", ["All", "task", "event"], horizontal=True)
    view = extract_df if type_filter == "All" else extract_df[extract_df["type"] == type_filter]
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("Items with unresolved fields")
    st.caption("These show the pipeline is honest about missing information rather than guessing it.")
    unresolved = extract_df[
        (extract_df["time"] == "unresolved")
        | (extract_df["person"] == "unresolved")
        | (extract_df["date_or_deadline"] == "unresolved")
    ]
    st.dataframe(unresolved.head(20), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
with tab3:
    st.subheader("Sensitive information detected")
    st.caption("All values shown below are already masked — raw sensitive values never appear in this app.")
    risk_filter = st.radio("Risk level", ["All", "high", "medium"], horizontal=True)
    view = sens_df if risk_filter == "All" else sens_df[sens_df["risk"] == risk_filter]
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("By sensitivity type")
    st.bar_chart(sens_df["sensitivity_type"].value_counts())

# ---------------------------------------------------------------
with tab4:
    st.subheader("TF-IDF embedding cross-check")
    st.caption(
        "An independent embedding-based validation layer on top of the rule-based classifier — "
        "TF-IDF vectors + cosine similarity to high-confidence rule-based anchor examples, run "
        "entirely locally (scikit-learn, no external API calls)."
    )
    c1, c2 = st.columns(2)
    c1.metric("Agreement rate (all messages)", f"{emb_summary['agreement_rate']:.1%}")
    c2.metric("Disagreements", emb_summary["disagreement_count"])

    st.info(emb_summary["note"])

    st.subheader("Rule vs. embedding disagreements")
    st.caption("Cases where the nearest embedding-neighbor category differs from the rule-based category — good candidates for manual review.")
    st.dataframe(
        pd.DataFrame(emb_disagreements).head(30),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------
with tab5:
    st.subheader("15 mandatory demo message IDs")
    st.dataframe(mand_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Built with a rule-based (regex) classification engine as the primary system, "
    "plus a local TF-IDF/embedding validation layer. Full source, README, and reasoning "
    "are in the linked GitHub repository."
)
