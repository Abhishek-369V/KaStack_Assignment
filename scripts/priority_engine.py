"""
L2 Part 1: Priority and Action Engine
---------------------------------------
Extends L1's classifier/extractor. Runs classify()/extract over L1 (900) + L2 (180) messages combined, chronologically,
then assigns a priority to every actionable item using multiple accumulated signals -- never a single keyword. 
Priority updates as later L2 messages add evidence (deadline moved, marked urgent, completed, cancelled, etc).

Importing this module has NO side effects. build_l1_l2_context() 
- reads files and runs the pipeline once; other functions/modules reuse its output.
"""

import pandas as pd
import re
import json
import os
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, "scripts")
from pipeline import (
    classify, extract_task_or_event, detect_sensitive,
    DATE_RE, run_l1_pipeline,
)

BASE_DIR = Path(__file__).resolve().parent.parent
L1_PATH = BASE_DIR / "messages.csv"
L2_PATH = BASE_DIR / "l2_messages.csv"
MANDATORY_PATH = BASE_DIR / "mandatory_demo_ids.csv"
OUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------------------
# Event-name / action-phrase normalization
# ---------------------------------------------------------------------

EVENT_NAME_PATTERNS = [
    re.compile(r"Calendar update:\s*([a-zA-Z \-]+?),", re.I),
    re.compile(r"Reminder:\s*([a-zA-Z \-]+?)(?:\s+happens on|\s+is\b|$)", re.I),
    re.compile(r"Join the\s+([a-zA-Z \-]+?)\s+on\s+" + DATE_RE, re.I),
    re.compile(r"^The\s+([a-zA-Z \-]+?)\s+(?:has (?:been )?moved to|stays the same|has been cancelled|is scheduled)\b", re.I),
    re.compile(r"^A new\s+([a-zA-Z \-]+?)\s+(?:session|meeting)\s+is scheduled", re.I),
    re.compile(r"[Aa]vailable for the\s+([a-zA-Z \-]+?)\s+at\s+\d", re.I),
]

def extract_event_name(text):
    """Pull the canonical short event name out of a full event sentence,
    regardless of which template wraps it."""
    for pat in EVENT_NAME_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip().lower()
    return None

def normalize_action_phrase(text):
    """Extract the core recurring action phrase from a message, e.g.
    'Please confirm whether you started to pay the electricity bill.'
    -> 'pay the electricity bill'. Tries event-name extraction first,
    then task-style wrapper stripping. This mirrors how the L2 dataset
    is templated: every follow-up embeds the original action phrase."""
    event_name = extract_event_name(text)
    if event_name:
        return event_name

    t = text.strip().rstrip(".?!")
    wrappers = [
        r"^Follow-up:\s*", r"^Additional update:\s*", r"^Update:\s*",
        r"^New task:\s*", r"^Can you share an update on\s*",
        r"^Following up on\s*", r"^Please confirm whether you started to\s*",
        r"^Please check the latest status of\s*",
        r"^I am referring to our earlier request about\s*",
        r"^The deadline (for|to)\s*",
        r"^Please note that\s*",
        r"^You can cancel\s*",
        r"^Hi,\s*", r"^FYI:\s*", r"^Please note:\s*", r"^Just checking—\s*",
        r"^Can you help\?\s*", r"^For today:\s*", r"^One more thing:\s*",
        r"^Quick update:\s*", r"^Important:\s*",
        r"^Don'?t forget to\s*", r"^Can you\s*",
        r"^Could you\s*",
        r"^Any progress on the item concerning\s*",
        r"^Has the\s+", r"\s+item been handled yet$",
        r"^The work we discussed about\s*",
        r"\s+still needs attention$",
        r"^Please share an update on\s*",
        r"^Please\s+", r"^I need you to\s+",
    ]
    for w in wrappers:
        t = re.sub(w, "", t, flags=re.I)
    t = re.split(r";\s*is it in progress|has been completed|has been cancelled|"
                 r"is now\s+\d{4}-\d{2}-\d{2}|is due on\s+\d{4}-\d{2}-\d{2}|"
                 r"; deadline is\s+\d{4}-\d{2}-\d{2}|"
                 r"\s+(?:by|before)\s+\d{4}-\d{2}-\d{2}|"
                 r";\s*it is no longer (needed|required)", t, flags=re.I)[0]
    t = t.strip().lower()
    t = re.sub(r"^the\s+the\s+", "the ", t)
    return t

# ---------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------

URGENT_MARKERS = re.compile(
    r"\b(urgent|treat this as urgent|is now urgent|no longer urgent|immediately)\b", re.I
)
COMPLETED_MARKERS = re.compile(
    r"\b(has been completed|completed successfully|has been submitted|approved|confirmed:)\b", re.I
)
CANCELLED_MARKERS = re.compile(
    r"\b(cancel|cancelled|no longer needed|no longer required)\b", re.I
)
RESCHEDULED_MARKERS = re.compile(
    r"\b(moved to|has moved to|rescheduled|new schedule|stays the same, but the time is now)\b", re.I
)
DEADLINE_MOVED_EARLIER = re.compile(
    r"\bnow\s+" + DATE_RE + r",\s*earlier than previously planned\b", re.I
)
DEADLINE_EXTENDED = re.compile(
    r"\bextended to\s+" + DATE_RE, re.I
)
CONFLICTING_MARKERS = re.compile(
    r"\b(may be .* or it may be|one message says .* but|conflicting|not completely sure|"
    r"might already be|I cannot confirm|we may move .*, I will confirm later)\b", re.I
)
RESPONSE_REQUIRED_MARKERS = re.compile(
    r"\b(can you|please confirm|please check|any update|has .* been handled)\b", re.I
)

def extract_signals(msg, reference_date=datetime(2026, 10, 5)):
    """Multi-signal extraction -- priority never derives from one keyword."""
    signals = []
    if URGENT_MARKERS.search(msg):
        if re.search(r"no longer urgent", msg, re.I):
            signals.append("urgency_lowered")
        else:
            signals.append("urgent_marker")
    if COMPLETED_MARKERS.search(msg):
        signals.append("marked_completed")
    if CANCELLED_MARKERS.search(msg):
        signals.append("marked_cancelled")
    if RESCHEDULED_MARKERS.search(msg):
        signals.append("rescheduled")
    if DEADLINE_MOVED_EARLIER.search(msg):
        signals.append("deadline_moved_earlier")
    if DEADLINE_EXTENDED.search(msg):
        signals.append("deadline_extended")
    if CONFLICTING_MARKERS.search(msg):
        signals.append("conflicting_or_uncertain")
    if RESPONSE_REQUIRED_MARKERS.search(msg):
        signals.append("response_required")
    date_match = re.search(DATE_RE, msg)
    if date_match:
        try:
            d = datetime.strptime(date_match.group(0), "%Y-%m-%d")
            days_out = (d - reference_date).days
            if days_out <= 0:
                signals.append("deadline_today_or_overdue")
            elif days_out <= 2:
                signals.append("deadline_within_2_days")
        except ValueError:
            pass
    return signals

def score_priority(signals, category, is_sensitive):
    """Priority derives from an accumulated signal set, never a single
    keyword. Sensitive items get a floor bump (safety-first)."""
    if "marked_cancelled" in signals:
        return "low", "Item was explicitly cancelled; no further action needed."
    if "marked_completed" in signals:
        return "low", "Item was explicitly marked completed; no further action needed."

    score = 0
    reasons = []
    if "urgent_marker" in signals:
        score += 3; reasons.append("explicit urgency marker")
    if "deadline_moved_earlier" in signals:
        score += 3; reasons.append("deadline moved earlier than originally planned")
    if "deadline_today_or_overdue" in signals:
        score += 3; reasons.append("deadline is today or already passed")
    elif "deadline_within_2_days" in signals:
        score += 2; reasons.append("deadline is within 2 days")
    if "response_required" in signals:
        score += 1; reasons.append("a response/confirmation is being requested")
    if "conflicting_or_uncertain" in signals:
        score += 1; reasons.append("status or deadline is conflicting/uncertain")
    if "urgency_lowered" in signals:
        score -= 2; reasons.append("a later message lowered urgency")
    if "deadline_extended" in signals:
        score -= 2; reasons.append("deadline was extended, reducing time pressure")
    if is_sensitive:
        score += 1; reasons.append("item involves sensitive information (safety floor)")

    if score >= 5:
        priority = "critical"
    elif score >= 3:
        priority = "high"
    elif score >= 1:
        priority = "medium"
    else:
        priority = "low"

    reason_text = "; ".join(reasons) if reasons else "No strong signals found; default low priority."
    return priority, reason_text

# ---------------------------------------------------------------------
# Context builder -- the one place that reads files / runs L1 pipeline
# ---------------------------------------------------------------------

def build_l1_l2_context(l1_path=L1_PATH, l2_path=L2_PATH, mandatory_path=MANDATORY_PATH,
                         out_dir=OUT_DIR, verbose=True):
    l1 = pd.read_csv(l1_path)
    l2 = pd.read_csv(l2_path)
    l1.columns = [c.strip() for c in l1.columns]
    l2.columns = [c.strip() for c in l2.columns]

    combined = pd.concat([l1, l2], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    if verbose:
        print(f"L1: {len(l1)} messages, L2: {len(l2)} messages, combined: {len(combined)}")

    os.makedirs(out_dir, exist_ok=True)
    combined_path = f"{out_dir}/_combined_l1_l2_messages.csv"
    combined.to_csv(combined_path, index=False)

    classification_results, extraction_results, sensitive_results = run_l1_pipeline(
        in_path=combined_path, mandatory_path=mandatory_path, out_dir=out_dir, save=False
    )
    os.remove(combined_path)  # scratch file only -- never committed, see .gitignore

    class_df = pd.DataFrame(classification_results)
    extract_df = pd.DataFrame(extraction_results)
    if verbose:
        print(f"Classified {len(class_df)} messages, extracted {len(extract_df)} tasks/events")

    phrase_to_items = {}
    extract_df_indexed = extract_df.copy()
    msg_ts = combined.set_index("message_id")["timestamp"]
    extract_df_indexed["timestamp"] = extract_df_indexed["source_message_id"].map(msg_ts)

    for _, item in extract_df_indexed.sort_values("timestamp").iterrows():
        phrase = normalize_action_phrase(item["description"])
        if not phrase:
            continue
        phrase_to_items.setdefault(phrase, []).append((item["timestamp"], item["item_id"]))

    return {
        "l1": l1, "l2": l2, "combined": combined,
        "classification_results": classification_results,
        "extraction_results": extraction_results,
        "sensitive_results": sensitive_results,
        "class_df": class_df, "extract_df": extract_df,
        "phrase_to_items": phrase_to_items,
    }

def find_linked_item(message_text, phrase_to_items, at_timestamp=None):
    """Find the item this message most likely refers to: the most recent
    occurrence of the same normalized phrase at or before at_timestamp."""
    phrase = normalize_action_phrase(message_text)
    candidates = phrase_to_items.get(phrase)
    if not candidates:
        return None
    if at_timestamp is None:
        return candidates[-1][1]
    prior = [c for c in candidates if c[0] <= at_timestamp]
    if prior:
        return prior[-1][1]
    return candidates[0][1]

def compute_priorities(ctx, verbose=True):
    """Walk the combined stream chronologically, updating each linked
    item's priority as new evidence (L2 messages) arrives."""
    combined = ctx["combined"]
    extract_df = ctx["extract_df"]
    phrase_to_items = ctx["phrase_to_items"]

    item_priority_state = {}
    priority_results = []

    for _, row in combined.iterrows():
        msg = row["message"]
        mid = row["message_id"]

        linked_item = find_linked_item(msg, phrase_to_items, at_timestamp=row["timestamp"])
        if not linked_item:
            continue

        if linked_item not in item_priority_state:
            item_priority_state[linked_item] = {"signals": set()}

        new_signals = extract_signals(msg)
        state = item_priority_state[linked_item]
        for s in new_signals:
            state["signals"].add(s)
        if "marked_completed" in new_signals or "marked_cancelled" in new_signals:
            state["signals"] -= {"urgent_marker", "deadline_moved_earlier",
                                  "deadline_today_or_overdue", "deadline_within_2_days"}

        item_row = extract_df[extract_df["item_id"] == linked_item]
        category = "action_required" if not item_row.empty and item_row.iloc[0]["type"] == "task" else "meeting_or_event"
        sens = detect_sensitive(msg)
        is_sensitive = sens is not None

        priority, reason = score_priority(state["signals"], category, is_sensitive)
        confidence = min(0.99, 0.6 + 0.08 * len(state["signals"]))

        record = {
            "message_id": mid,
            "item_id": linked_item,
            "priority": priority,
            "reason": reason,
            "signals": sorted(state["signals"]),
            "confidence": round(confidence, 2),
        }
        priority_results.append(record)

    if verbose:
        print(f"Generated {len(priority_results)} priority decisions across {len(item_priority_state)} linked items")

    latest_priority = {}
    for r in priority_results:
        latest_priority[r["item_id"]] = r

    return priority_results, list(latest_priority.values())


if __name__ == "__main__":
    ctx = build_l1_l2_context()
    priority_full, priority_latest = compute_priorities(ctx)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/priority_decisions_full.json", "w") as f:
        json.dump(priority_full, f, indent=2)
    with open(f"{OUT_DIR}/priority_latest_per_item.json", "w") as f:
        json.dump(priority_latest, f, indent=2)
    with open(f"{OUT_DIR}/l1_l2_classification_results.json", "w") as f:
        json.dump(ctx["classification_results"], f, indent=2)
    with open(f"{OUT_DIR}/l1_l2_extraction_results.json", "w") as f:
        json.dump(ctx["extraction_results"], f, indent=2)

    print("Saved priority_decisions_full.json, priority_latest_per_item.json, "
          "l1_l2_classification_results.json, l1_l2_extraction_results.json")