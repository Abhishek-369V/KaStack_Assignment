"""
L2 Part 3: Semantic Search and Intelligent Assistant
---------------------------------------------------------
Extends L1+L2 with semantic search and question answering, using L1's classifications, extracted tasks/events, 
sensitive-info results, Part 1's priority results, and Part 2's related-message groups.
Retrieval method: TF-IDF (1-2 grams) + cosine similarity over message text -- consistent with L1's own 
embedding-validation approach (same philosophy: transparent, explainable, no external API calls, no fabricated 
benchmarks). 
This is the "baseline" retrieval method; an "optimized" variant (dimensionality-reduced via TruncatedSVD) is also
built for the required benchmark comparison in the README/video.

The assistant answers a fixed set of question types drawn directly from the assignment brief. 
It NEVER answers if there isn't supporting evidence -- it returns an explicit "insufficient evidence" response
instead of guessing.
"""

import pandas as pd
import re
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
from priority_engine import build_l1_l2_context, find_linked_item
from grouping_engine import build_groups

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"


class SemanticIndex:
    """Wraps a TF-IDF (optionally SVD-reduced) index over the combined
    L1+L2 message set, plus lookups into priority/group/sensitive data so
    answers can cite supporting evidence with IDs, not just raw text."""

    def __init__(self, ctx, groups, priority_latest, use_svd=False, svd_components=100):
        self.ctx = ctx
        self.combined = ctx["combined"].reset_index(drop=True)
        self.class_df = ctx["class_df"]
        self.extract_df = ctx["extract_df"]
        self.sensitive_results = ctx["sensitive_results"]
        self.groups = groups
        self.priority_latest = {p["item_id"]: p for p in priority_latest}
        self.use_svd = use_svd

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        tfidf_matrix = self.vectorizer.fit_transform(self.combined["message"])

        if use_svd:
            n_comp = min(svd_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self.matrix = self.svd.fit_transform(tfidf_matrix)
        else:
            self.svd = None
            self.matrix = tfidf_matrix

        # quick lookups
        self.mid_to_row = {r["message_id"]: i for i, r in self.combined.iterrows()}
        self.item_by_id = {r["item_id"]: r for r in self.extract_df.to_dict(orient="records")}
        self.class_by_mid = {r["message_id"]: r for r in self.class_df.to_dict(orient="records")}
        self.sensitive_by_mid = {r["message_id"]: r for r in self.sensitive_results}

    def _vectorize_query(self, query):
        q_tfidf = self.vectorizer.transform([query])
        if self.use_svd:
            return self.svd.transform(q_tfidf)
        return q_tfidf

    def search(self, query, top_k=10):
        q_vec = self._vectorize_query(query)
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            row = self.combined.iloc[idx]
            results.append({
                "message_id": row["message_id"],
                "message": row["message"],
                "relevance_score": round(float(sims[idx]), 4),
            })
        return results

    def index_size_bytes(self):
        if self.use_svd:
            return self.matrix.nbytes
        return self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes


# ---------------------------------------------------------------------
# Privacy-aware routing (required by video demo checklist: one request
# processed locally, one requiring confirmation, one blocked)
# ---------------------------------------------------------------------

def privacy_route(query, index):
    """Decides whether a query can be answered locally, needs
    confirmation, or must be blocked, based on whether it touches
    sensitive-information message content."""
    q_lower = query.lower()
    sensitive_keywords = ["password", "otp", "token", "card number", "account number",
                           "recovery code", "identification number"]

    if any(k in q_lower for k in sensitive_keywords):
        return "blocked", "Query directly requests sensitive credential/financial content; refusing to surface even masked values outside their designated review flow."

    # if the query's top retrieval hit is itself a sensitive-info message,
    # require confirmation before returning anything derived from it
    results = index.search(query, top_k=3)
    for r in results:
        if r["message_id"] in index.sensitive_by_mid:
            return "confirm", f"Top matching evidence ({r['message_id']}) is flagged sensitive_information; showing masked summary only after explicit confirmation."

    return "local", "No sensitive-information content involved; safe to process and answer locally."


# ---------------------------------------------------------------------
# Question answering: fixed set of question types from the assignment
# brief, each backed by real evidence lookups (priority/groups/class/
# extraction), never a free-form generative answer.
# ---------------------------------------------------------------------

def answer_query(query, index):
    q = query.strip().lower()
    route, route_reason = privacy_route(query, index)

    if route == "blocked":
        return {
            "query": query,
            "answer": "This query touches sensitive credential/financial content and cannot be answered directly.",
            "supporting_message_ids": [],
            "group_id": None,
            "reason": route_reason,
            "privacy_route": route,
        }

    # --- "What tasks should I complete today?" / "critical or high priority" ---
    if "today" in q or ("critical" in q or "high" in q or "priority" in q):
        wanted = None
        if "critical" in q:
            wanted = {"critical"}
        elif "high" in q:
            wanted = {"high", "critical"}

        candidates = []
        for item_id, p in index.priority_latest.items():
            if wanted and p["priority"] not in wanted:
                continue
            if "today" in q and "deadline_today_or_overdue" not in p.get("signals", []):
                continue
            candidates.append(p)

        if not candidates:
            return {
                "query": query, "answer": "No matching pending items found with sufficient evidence.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No priority record matched the requested filter (today / critical / high).",
                "privacy_route": route,
            }

        item_titles = []
        supporting_ids = []
        for p in candidates[:5]:
            item = index.item_by_id.get(p["item_id"])
            title = item["title"] if item else p["item_id"]
            item_titles.append(f"{p['item_id']} ({p['priority']}): {title}")
            supporting_ids.append(p["message_id"])

        return {
            "query": query,
            "answer": "; ".join(item_titles),
            "supporting_message_ids": supporting_ids,
            "group_id": None,
            "reason": f"Filtered {len(candidates)} priority record(s) matching the requested urgency/date filter; showing up to 5.",
            "privacy_route": route,
        }

    # --- "Show all messages related to X" ---
    m = re.search(r"related to (.+?)\??$", q)
    if m or "related to" in q:
        subject = (m.group(1) if m else q.split("related to")[-1]).strip()
        best_group, best_score = None, 0
        for g in index.groups:
            title_words = set(g["title"].lower().split())
            subject_words = set(subject.split())
            overlap = len(title_words & subject_words)
            if overlap > best_score:
                best_score = overlap
                best_group = g
        if not best_group or best_score == 0:
            return {
                "query": query, "answer": "No related-message group found for this subject.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No group title overlapped with the requested subject terms.",
                "privacy_route": route,
            }
        return {
            "query": query,
            "answer": best_group["summary"],
            "supporting_message_ids": best_group["related_message_ids"],
            "group_id": best_group["group_id"],
            "reason": f"Matched group '{best_group['title']}' (status: {best_group['status']}) by subject-term overlap.",
            "privacy_route": route,
        }

    # --- "What meetings were rescheduled?" ---
    if "reschedul" in q:
        matches = [g for g in index.groups if g["status"] == "rescheduled"]
        if not matches:
            return {
                "query": query, "answer": "No rescheduled meetings found with sufficient evidence.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No group's status resolved to 'rescheduled'.",
                "privacy_route": route,
            }
        answer = "; ".join(f"{g['title']} (group {g['group_id']})" for g in matches[:5])
        all_ids = [mid for g in matches[:5] for mid in g["related_message_ids"]]
        return {
            "query": query, "answer": answer, "supporting_message_ids": all_ids,
            "group_id": matches[0]["group_id"],
            "reason": f"{len(matches)} group(s) resolved to status 'rescheduled'.",
            "privacy_route": route,
        }

    # --- "Which tasks have been completed?" ---
    if "completed" in q:
        matches = [g for g in index.groups if g["status"] == "completed"]
        if not matches:
            return {
                "query": query, "answer": "No completed items found with sufficient evidence.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No group's status resolved to 'completed'.",
                "privacy_route": route,
            }
        answer = "; ".join(f"{g['title']} (group {g['group_id']})" for g in matches[:5])
        all_ids = [mid for g in matches[:5] for mid in g["related_message_ids"]]
        return {
            "query": query, "answer": answer, "supporting_message_ids": all_ids,
            "group_id": matches[0]["group_id"],
            "reason": f"{len(matches)} group(s) resolved to status 'completed'.",
            "privacy_route": route,
        }

    # --- "What is the latest status of X?" ---
    if "latest status" in q or "status of" in q:
        best_group, best_score = None, 0
        for g in index.groups:
            title_words = set(g["title"].lower().split())
            q_words = set(q.split())
            overlap = len(title_words & q_words)
            if overlap > best_score:
                best_score = overlap
                best_group = g
        if not best_group:
            return {
                "query": query, "answer": "Insufficient evidence to determine status for this item.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No group title overlapped with the query subject.",
                "privacy_route": route,
            }
        return {
            "query": query,
            "answer": f"Status: {best_group['status']}. {best_group['summary']}",
            "supporting_message_ids": best_group["related_message_ids"],
            "group_id": best_group["group_id"],
            "reason": f"Matched group '{best_group['title']}'; status reflects the most recent message in the group.",
            "privacy_route": route,
        }

    # --- "Which messages require confirmation?" ---
    if "require confirmation" in q or "requires confirmation" in q:
        matches = [p for p in index.priority_latest.values() if "response_required" in p.get("signals", [])]
        if not matches:
            return {
                "query": query, "answer": "No messages currently require confirmation.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No priority record has the 'response_required' signal.",
                "privacy_route": route,
            }
        ids = [p["message_id"] for p in matches[:10]]
        return {
            "query": query,
            "answer": f"{len(matches)} item(s) require confirmation.",
            "supporting_message_ids": ids, "group_id": None,
            "reason": "Matched priority records with the 'response_required' signal.",
            "privacy_route": route,
        }

    # --- "Why was this item marked as critical?" ---
    if "why" in q and "critical" in q:
        crit = [p for p in index.priority_latest.values() if p["priority"] == "critical"]
        if not crit:
            return {
                "query": query, "answer": "No items are currently marked critical.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No priority record resolved to 'critical'.",
                "privacy_route": route,
            }
        p = crit[0]
        return {
            "query": query, "answer": p["reason"],
            "supporting_message_ids": [p["message_id"]], "group_id": None,
            "reason": f"Reason taken directly from the priority decision for {p['item_id']}.",
            "privacy_route": route,
        }

    # --- "What deadlines have changed?" ---
    if "deadline" in q and "chang" in q:
        matches = [p for p in index.priority_latest.values()
                   if "deadline_moved_earlier" in p.get("signals", []) or "deadline_extended" in p.get("signals", [])]
        if not matches:
            return {
                "query": query, "answer": "No deadline changes found with sufficient evidence.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No priority record has a deadline-change signal.",
                "privacy_route": route,
            }
        ids = [p["message_id"] for p in matches[:10]]
        return {
            "query": query,
            "answer": f"{len(matches)} item(s) had a deadline change (moved earlier or extended).",
            "supporting_message_ids": ids, "group_id": None,
            "reason": "Matched priority records with deadline_moved_earlier or deadline_extended signals.",
            "privacy_route": route,
        }

    # --- "Are there any conflicting messages about the same event?" ---
    if "conflict" in q:
        matches = [g for g in index.groups if g["status"] == "unclear"]
        if not matches:
            return {
                "query": query, "answer": "No conflicting/uncertain message groups found.",
                "supporting_message_ids": [], "group_id": None,
                "reason": "No group's status resolved to 'unclear'.",
                "privacy_route": route,
            }
        answer = "; ".join(f"{g['title']} (group {g['group_id']})" for g in matches[:5])
        all_ids = [mid for g in matches[:5] for mid in g["related_message_ids"]]
        return {
            "query": query, "answer": answer, "supporting_message_ids": all_ids,
            "group_id": matches[0]["group_id"],
            "reason": f"{len(matches)} group(s) resolved to status 'unclear' due to conflicting signals.",
            "privacy_route": route,
        }

    # --- fallback: pure semantic search, evidence-gated ---
    results = index.search(query, top_k=5)
    strong_results = [r for r in results if r["relevance_score"] > 0.15]
    if not strong_results:
        return {
            "query": query,
            "answer": "Insufficient evidence available to answer this query confidently.",
            "supporting_message_ids": [], "group_id": None,
            "reason": "No retrieved message exceeded the minimum relevance threshold (0.15).",
            "privacy_route": route,
        }
    return {
        "query": query,
        "answer": f"Most relevant message: {strong_results[0]['message_id']}",
        "supporting_message_ids": [r["message_id"] for r in strong_results],
        "group_id": None,
        "reason": "Fell back to semantic similarity search; no fixed question-type pattern matched.",
        "retrieved_relevance_scores": [r["relevance_score"] for r in strong_results],
        "privacy_route": route,
    }


# ---------------------------------------------------------------------
# Benchmark: baseline TF-IDF vs SVD-reduced ("optimized") retrieval
# ---------------------------------------------------------------------

def run_benchmark(ctx, groups, priority_latest, sample_queries):
    results = {}
    for label, use_svd in [("baseline_tfidf", False), ("optimized_svd", True)]:
        t0 = time.time()
        index = SemanticIndex(ctx, groups, priority_latest, use_svd=use_svd)
        build_time = time.time() - t0

        t0 = time.time()
        for q in sample_queries:
            index.search(q, top_k=5)
        query_time = (time.time() - t0) / max(len(sample_queries), 1)

        results[label] = {
            "build_time_seconds": round(build_time, 4),
            "avg_query_time_seconds": round(query_time, 6),
            "index_size_bytes": int(index.index_size_bytes()),
            "index_size_kb": round(index.index_size_bytes() / 1024, 2),
        }
    return results


if __name__ == "__main__":
    print("Building L1+L2 context...")
    ctx = build_l1_l2_context()

    print("Computing priorities...")
    from priority_engine import compute_priorities
    priority_full, priority_latest = compute_priorities(ctx)

    print("Building related-message groups...")
    groups = build_groups(ctx)

    print("Building semantic index (baseline TF-IDF)...")
    index = SemanticIndex(ctx, groups, priority_latest, use_svd=False)

    # demo queries drawn directly from the assignment brief
    demo_queries = [
        "What tasks should I complete today?",
        "Which critical or high-priority tasks are still pending?",
        "Show all messages related to the project report",
        "What meetings were rescheduled?",
        "Which tasks have been completed?",
        "What is the latest status of this task?",
        "Which messages require confirmation?",
        "Why was this item marked as critical?",
        "What deadlines have changed?",
        "Are there any conflicting messages about the same event?",
    ]

    answers = [answer_query(q, index) for q in demo_queries]

    with open(f"{OUT_DIR}/semantic_qa_results.json", "w") as f:
        json.dump(answers, f, indent=2)
    print(f"Saved {OUT_DIR}/semantic_qa_results.json")

    # privacy-routing demo file (one local, one confirm, one blocked example)
    privacy_examples = [
        answer_query("What tasks should I complete today?", index),          # local
        answer_query("Show all messages related to the OTP", index),         # likely confirm/blocked
        answer_query("What is the password for the demo account?", index),   # blocked
    ]
    with open(f"{OUT_DIR}/privacy_routing_results.json", "w") as f:
        json.dump(privacy_examples, f, indent=2)
    print(f"Saved {OUT_DIR}/privacy_routing_results.json")

    print("Running benchmark (baseline vs optimized)...")
    benchmark = run_benchmark(ctx, groups, priority_latest, demo_queries)
    with open(f"{OUT_DIR}/benchmark_comparison.json", "w") as f:
        json.dump(benchmark, f, indent=2)
    print(json.dumps(benchmark, indent=2))
    print(f"Saved {OUT_DIR}/benchmark_comparison.json")