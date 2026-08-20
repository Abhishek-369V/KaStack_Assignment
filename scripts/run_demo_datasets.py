"""
Run the full L2 system against the supplied L2 demo dataset and demo
queries -- these are the mandatory test cases referenced in the L2 Assignment.
"""

import json
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from priority_engine import build_l1_l2_context, compute_priorities, find_linked_item
from grouping_engine import build_groups
from semantic_assistant import SemanticIndex, answer_query

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"

if __name__ == "__main__":
    ctx = build_l1_l2_context()
    priority_full, priority_latest = compute_priorities(ctx)
    groups = build_groups(ctx)
    index = SemanticIndex(ctx, groups, priority_latest, use_svd=False)

    # --- demo messages: show classification + priority for each ---
    demo_msgs = pd.read_csv(BASE_DIR / "l2_demo_messages.csv")
    demo_msgs.columns = [c.strip() for c in demo_msgs.columns]

    class_lookup = {r["message_id"]: r for r in ctx["classification_results"]}
    priority_lookup = {p["message_id"]: p for p in priority_full}

    demo_results = []
    for _, row in demo_msgs.iterrows():
        mid = row["message_id"]
        entry = {
            "message_id": mid,
            "message": row["message"],
            "classification": class_lookup.get(mid),
            "priority": priority_lookup.get(mid),
        }
        demo_results.append(entry)

    with open(f"{OUT_DIR}/mandatory_demo_messages_results.json", "w") as f:
        json.dump(demo_results, f, indent=2)
    print(f"Saved {OUT_DIR}/mandatory_demo_messages_results.json ({len(demo_results)} messages)")

    # --- demo queries: run through the assistant ---
    demo_queries = pd.read_csv(BASE_DIR / "l2_demo_queries.csv")
    demo_queries.columns = [c.strip() for c in demo_queries.columns]
    query_col = "query"

    query_results = []
    for _, row in demo_queries.iterrows():
        q = row[query_col]
        result = answer_query(q, index)
        result["query_id"] = row["query_id"]
        query_results.append(result)

    with open(f"{OUT_DIR}/mandatory_demo_queries_results.json", "w") as f:
        json.dump(query_results, f, indent=2)
    print(f"Saved {OUT_DIR}/mandatory_demo_queries_results.json ({len(query_results)} queries)")