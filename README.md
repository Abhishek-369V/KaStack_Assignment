# Message Intelligence Pipeline — L1 + L2

A rule-based system that classifies messages, extracts tasks/events, detects
and masks sensitive information, then (L2) assigns priority, groups related
messages over time, and answers questions through a semantic assistant.

**Repo layout**: this is a single project. L2 does not duplicate L1 — every
L2 script imports directly from `scripts/pipeline.py`, the same file used
for the original L1 submission. Nothing in L1 was rewritten from scratch for
L2; L2 only adds new modules on top of it (see `scripts/` below).

```
notebook/       — L1's annotated notebook (classification, extraction, sensitive detection, embedding validation)
scripts/        — pipeline.py (L1 core, imported by everything below) + priority_engine.py, grouping_engine.py, semantic_assistant.py, run_demo_datasets.py (all L2)
streamlit_app/  — single cloud demo app covering both L1 and L2 views
outputs/        — all generated JSON output files (L1-only + L1+L2 combined)
```

---

## Part 0: How L2 extends L1 (not a rebuild)

L2's every module (`priority_engine.py`, `grouping_engine.py`,
`semantic_assistant.py`) imports `classify()`, `extract_task_or_event()`,
`detect_sensitive()`, and `run_l1_pipeline()` directly from `pipeline.py` —
the same functions that produced the original L1 submission. `pipeline.py`
was refactored from a standalone script into an importable module (wrapped
in `run_l1_pipeline()`, guarded by `if __name__ == "__main__"`) specifically
so L2 could reuse it without copy-pasting logic or re-deriving classification
rules. L2 runs this same pipeline over the **combined L1 (900) + L2 (180) =
1080 messages**, in chronological order, then layers three new capabilities
on top of that single combined classification/extraction pass.

---

## Part 1: Priority and Action Engine

Every task/event extracted by L1's logic gets a priority — `critical`,
`high`, `medium`, or `low` — derived from an **accumulated set of signals**,
never a single keyword:

- explicit urgency markers ("urgent", "treat this as urgent")
- deadline proximity (today/overdue, within 2 days) and deadline *changes*
  (moved earlier, extended)
- whether a response/confirmation is being requested
- whether the item is sensitive (safety floor bump)
- explicit completion or cancellation (forces priority to `low`)
- conflicting/uncertain status language

**Linking L2 follow-ups back to their L1 origin**: L2 messages ("Follow-up:
X is now urgent", "X has been completed") don't repeat the full original
message — they reference it by a shortened phrase. `normalize_action_phrase()`
strips known wrapper prefixes/suffixes (from both L1's and L2's message
templates) to recover a canonical phrase, e.g. both *"Please pay the
electricity bill by 2026-09-04"* and *"Follow-up: pay the electricity bill;
is it in progress?"* normalize to `"pay the electricity bill"`. Messages are
indexed by this phrase, **timestamp-sorted**, so a follow-up always links to
the most recent prior occurrence of that phrase — this matters because some
phrases recur multiple times in L1 with different deadlines (e.g. a
recurring "upload the assignment" task appears 10 separate times), and a
follow-up should resolve to whichever occurrence is chronologically nearest,
not always the first.

**Priority updates over time**: as more L2 messages reference the same item,
their signals accumulate onto that item's running state. A later
"completed"/"cancelled" message clears any earlier urgency signals and forces
priority to `low`, since those are treated as terminal states.

Current output: 378 linked items, 486 total priority decisions (some items
receive multiple decisions as new evidence arrives) →
`outputs/priority_decisions_full.json` (full history) and
`outputs/priority_latest_per_item.json` (final state per item).

---

## Part 2: Related-Message Grouping

Reuses Part 1's phrase-based linkage as the grouping key, rather than a
second independent similarity heuristic — two messages Part 1 already
resolved to the same `item_id` are by definition the same subject, so using
a different grouping method could produce contradictory results between the
two parts.

For each group:
- **status** (`pending` / `in_progress` / `completed` / `rescheduled` /
  `cancelled` / `unclear`) is set by the **most recent relevant signal** in
  the group's messages, walked chronologically — not a majority vote, since
  a single later "completed" message should override several earlier
  "pending" ones.
- **latest_deadline** is the date mentioned in the chronologically **latest**
  message that has one — not the largest date value, since a later message
  can move a deadline earlier.
- **summary** is built only from counts and status of the actual messages
  (e.g. "5 message(s) track 'pay the electricity bill'. The deadline was
  mentioned or changed more than once...") — no fabricated narrative detail.

Current output: 44 groups (2+ messages each) →
`outputs/related_message_groups.json`. Status distribution: 14 pending, 13
rescheduled, 6 in_progress, 6 cancelled, 5 completed.

---

## Part 3: Semantic Search and Intelligent Assistant

**Retrieval**: TF-IDF (1-2 grams) + cosine similarity over the combined
message set — the same approach as L1's own embedding-validation layer, kept
consistent for the same reason: transparent, fully local, no external API
calls (satisfying "do not send raw messages to external AI services").

**Question answering**: the assistant handles the ten question types listed
in the assignment brief via dedicated logic that queries Part 1's priority
data and Part 2's groups directly (e.g. "which tasks are still pending" →
filters `priority_latest_per_item.json`; "what meetings were rescheduled" →
filters groups by `status == "rescheduled"`). Only the fallback path (no
fixed question type matched) uses raw semantic similarity search.

**The assistant never fabricates an answer.** If no priority record, group,
or retrieved message provides sufficient evidence (semantic similarity below
a 0.15 threshold, or no matching status/signal found), it returns "Insufficient
evidence available to answer this query confidently" along with the reason,
rather than guessing.

**Privacy-aware routing**: every query is classified as `local` (safe, no
sensitive content involved), `confirm` (top retrieved evidence is itself a
sensitive-information message — requires explicit confirmation before
returning anything derived from it), or `blocked` (query directly names a
credential/financial term like "password" or "OTP" — refused outright, no
masked value surfaced even). See `outputs/privacy_routing_results.json` for
one example of each.

**Benchmark (baseline vs. "optimized")**: baseline is plain TF-IDF; the
"optimized" variant applies TruncatedSVD dimensionality reduction on top.
Honest result — see `outputs/benchmark_comparison.json`:

| | build time | avg query time | index size |
|---|---|---|---|
| baseline (TF-IDF) | 0.120s | 0.0032s | 175 KB |
| "optimized" (SVD) | 0.383s | 0.0057s | 844 KB |

**The SVD variant is actually slower and larger, not faster/smaller, at this
dataset size.** This is a real result, not an error: SVD's fixed overhead
(matrix decomposition, denser output representation) dominates at ~1,080
messages. SVD's benefit (compact, fast retrieval) only shows up at a much
larger corpus size where the original sparse TF-IDF matrix becomes
genuinely large. Reported honestly rather than picking whichever number
looked better — this is exactly the trade-off the video's benchmark section
should show.

---

## Assumptions

- All L1 assumptions still apply (see below)
- A follow-up message's linked item is resolved by matching a normalized
  action phrase, not by any explicit ID reference in the dataset (none exists)
- `deadline_today_or_overdue` in the priority engine is computed against a
  fixed reference date (2026-10-05, chosen as the latest activity date across
  the combined dataset) — see Limitations below for why this affects priority
  distribution
- A "group" requires at least 2 related messages by definition; single
  unlinked messages (genuinely new L2 tasks, standalone sensitive
  disclosures, ambient/promotional messages) are correctly not grouped

## Limitations

**Priority skew**: 350 of 378 items (~93%) resolve to `high` priority,
because the fixed reference date (2026-10-05) falls after most of L1's
message dates (concentrated in September 2026), so the "deadline is today
or overdue" signal fires very broadly. This is disclosed rather than
silently adjusted — a synthetic dataset has no true "current date," and
picking a different reference date would just shift which messages trigger
the signal without resolving the underlying ambiguity. In a live system,
this reference date would be `datetime.now()`.

**SVD "optimization" underperforms baseline at this scale** — see benchmark
above. Reported honestly.

**Phrase-matching linkage is dataset-specific**: `normalize_action_phrase()`
strips wrapper phrases observed directly in this dataset's templates. It
would need generalizing (e.g. via fuzzy matching or embeddings) to handle
free-form messages that don't follow a template.

**Semantic search fallback threshold (0.15) is a manually chosen cutoff**,
not learned or validated against ground truth — reasonable for this dataset
size but arbitrary in principle.

All L1 limitations (see original notes) still apply: regex-based
classification is template-specific, `person` extraction is conservative,
some messages are genuinely ambiguous even to a human reader.

---

## AI-tool usage declaration

Claude (Anthropic) was used as a coding/debugging assistant throughout both
L1 and L2 — for exploring dataset structure, iterating on regex patterns
after reviewing false positives against real sample data, debugging the
phrase-linkage logic (multiple rounds of tracing why specific L2 follow-up
messages failed to link to their L1 origin, and fixing the root cause each
time), structuring this README, and styling the Streamlit frontend. All
logic was reviewed, tested against the actual dataset, and is understood by
me — I can explain and modify any part of it, including the priority scoring
weights, the phrase-normalization regex, the grouping status logic, and the
retrieval/benchmark approach. No message content was sent to any external AI
API as part of the actual classification/extraction/retrieval pipeline —
everything runs locally via Python, regex, pandas, and scikit-learn.

Given the scope of L2 (priority engine + grouping + semantic assistant +
privacy routing + benchmarking, on top of L1) within the assignment's time
window, this was built with heavy iterative AI assistance rather than solo
from scratch — disclosed here directly rather than implied otherwise.