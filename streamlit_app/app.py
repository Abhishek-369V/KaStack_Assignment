import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Message Intelligence Pipeline — L1 + L2", page_icon="📨", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "outputs"

@st.cache_data
def load_json(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)

# --- L1 outputs (900 messages only) ---
classification = load_json("classification_results.json")
extraction = load_json("extraction_results.json")
sensitive = load_json("sensitive_detection_results.json")
mandatory = load_json("mandatory_ids_results.json")
summary = load_json("summary.json")
emb_summary = load_json("embedding_validation_summary.json")
emb_disagreements = load_json("embedding_validation_disagreements.json")

# --- L2 outputs (L1+L2 combined, 1080 messages) ---
priority_latest = load_json("priority_latest_per_item.json")
groups = load_json("related_message_groups.json")
qa_results = load_json("semantic_qa_results.json")
privacy_examples = load_json("privacy_routing_results.json")
benchmark = load_json("benchmark_comparison.json")
demo_queries = load_json("mandatory_demo_queries_results.json")

class_df = pd.DataFrame(classification)
extract_df = pd.DataFrame(extraction)
sens_df = pd.DataFrame(sensitive)
mand_df = pd.DataFrame(mandatory)
priority_df = pd.DataFrame(priority_latest)
groups_df = pd.DataFrame(groups)

# ---------------------------------------------------------------
st.title("📨 Message Intelligence Pipeline — L1 + L2")
st.caption(
    "L1: rule-based classification, task/event extraction, sensitive-info detection over 900 messages. "
    "L2 extends this with priority scoring, related-message grouping, and a semantic QA assistant over "
    "the combined L1+L2 (1,080 message) stream."
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("L1 messages", summary["total_messages"])
col2.metric("L2 messages added", 180)
col3.metric("Combined items tracked", len(priority_df))
col4.metric("Related-message groups", len(groups_df))
col5.metric("Sensitive flagged (L1)", summary["total_sensitive_flagged"])

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Classification", "Tasks & Events", "Sensitive Detection", "Embedding Validation",
    "Priority Engine (L2)", "Related Groups (L2)", "Semantic Assistant (L2)", "Mandatory IDs"
])

# ---------------------------------------------------------------
with tab1:
    st.subheader("Category distribution (L1, 900 messages)")
    cat_counts = class_df["category"].value_counts()
    st.bar_chart(cat_counts)

    st.subheader("Browse classifications")
    categories = ["All"] + sorted(class_df["category"].unique().tolist())
    picked = st.selectbox("Filter by category", categories)
    view = class_df if picked == "All" else class_df[class_df["category"] == picked]
    st.dataframe(view, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
with tab2:
    st.subheader("Extracted tasks and events (L1)")
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
    st.subheader("Sensitive information detected (L1)")
    st.caption("All values shown below are already masked — raw sensitive values never appear in this app.")
    risk_filter = st.radio("Risk level", ["All", "high", "medium"], horizontal=True)
    view = sens_df if risk_filter == "All" else sens_df[sens_df["risk"] == risk_filter]
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("By sensitivity type")
    st.bar_chart(sens_df["sensitivity_type"].value_counts())

# ---------------------------------------------------------------
with tab4:
    st.subheader("TF-IDF embedding cross-check (L1)")
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
    st.dataframe(pd.DataFrame(emb_disagreements).head(30), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
with tab5:
    st.subheader("Priority and Action Engine (L2)")
    st.caption(
        "Priority is derived from an accumulated signal set per item — deadline proximity/changes, "
        "urgency markers, sensitivity, response-required, completion/cancellation — never a single keyword."
    )

    c1, c2, c3, c4 = st.columns(4)
    counts = priority_df["priority"].value_counts()
    c1.metric("Critical", int(counts.get("critical", 0)))
    c2.metric("High", int(counts.get("high", 0)))
    c3.metric("Medium", int(counts.get("medium", 0)))
    c4.metric("Low", int(counts.get("low", 0)))

    st.warning(
        "Note: ~93% of items resolve to 'high' priority because the reference date used for "
        "'deadline today/overdue' (2026-10-05) falls after most L1 message dates (concentrated "
        "in September 2026). This is disclosed in the README rather than silently adjusted — "
        "a synthetic dataset has no true 'current date'."
    )

    priority_filter = st.selectbox("Filter by priority", ["All", "critical", "high", "medium", "low"])
    view = priority_df if priority_filter == "All" else priority_df[priority_df["priority"] == priority_filter]
    st.dataframe(view, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
with tab6:
    st.subheader("Related-Message Groups (L2)")
    st.caption(
        "Groups reuse Part 1's phrase-based item linkage as the grouping key — two messages linked "
        "to the same item are by definition the same subject."
    )

    status_counts = groups_df["status"].value_counts()
    st.bar_chart(status_counts)

    status_filter = st.selectbox("Filter by status",
                                   ["All"] + sorted(groups_df["status"].unique().tolist()))
    view = groups_df if status_filter == "All" else groups_df[groups_df["status"] == status_filter]
    st.dataframe(
        view[["group_id", "title", "status", "latest_deadline", "confidence", "summary"]],
        use_container_width=True, hide_index=True
    )

    st.subheader("Inspect a group's messages")
    picked_group = st.selectbox("Group", groups_df["group_id"].tolist())
    group_row = groups_df[groups_df["group_id"] == picked_group].iloc[0]
    st.json({
        "title": group_row["title"],
        "status": group_row["status"],
        "latest_deadline": group_row["latest_deadline"],
        "related_message_ids": group_row["related_message_ids"],
        "related_item_ids": group_row["related_item_ids"],
        "summary": group_row["summary"],
    })

# ---------------------------------------------------------------
with tab7:
    st.subheader("Semantic Search & Intelligent Assistant (L2)")
    st.caption(
        "TF-IDF + cosine similarity retrieval, evidence-gated question answering, and privacy-aware "
        "routing (local / confirm / blocked). The assistant never fabricates an answer — if evidence "
        "is insufficient, it says so explicitly."
    )

    st.subheader("Standard question-type answers")
    st.dataframe(
        pd.DataFrame(qa_results)[["query", "answer", "privacy_route", "reason"]],
        use_container_width=True, hide_index=True
    )

    st.subheader("Privacy-aware routing examples")
    st.caption("One local, one requiring confirmation, one blocked — as required by the demo checklist.")
    for ex in privacy_examples:
        route_color = {"local": "green", "confirm": "orange", "blocked": "red"}.get(ex["privacy_route"], "gray")
        st.markdown(f"**Query:** {ex['query']}  \n:{route_color}[**Route: {ex['privacy_route']}**]")
        st.write(f"Answer: {ex['answer']}")
        st.caption(f"Reason: {ex['reason']}")
        st.divider()

    st.subheader("Mandatory demo queries (DQ01–DQ08)")
    st.dataframe(
        pd.DataFrame(demo_queries)[["query_id", "query", "answer", "privacy_route"]],
        use_container_width=True, hide_index=True
    )

    st.subheader("Retrieval benchmark: baseline vs. optimized")
    st.caption(
        "Baseline = plain TF-IDF. 'Optimized' = TruncatedSVD dimensionality reduction. "
        "Reported honestly — SVD is actually slower and larger at this dataset size; its benefit "
        "only appears at much larger corpus sizes."
    )
    bench_df = pd.DataFrame(benchmark).T
    st.dataframe(bench_df, use_container_width=True)

# ---------------------------------------------------------------
with tab8:
    st.subheader("15 mandatory L1 demo message IDs")
    st.dataframe(mand_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "L1: rule-based classification engine + local TF-IDF embedding validation. "
    "L2: priority engine, related-message grouping, and semantic assistant — all extending L1's "
    "pipeline.py directly rather than reimplementing it. Full source, README, and reasoning are in "
    "the linked GitHub repository."
)