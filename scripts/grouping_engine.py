"""
L2 Part 2: Related-Message Grouping
--------------------------------------
Groups messages referring to the same task/meeting/event/subject, reusing Part 1's item-linkage 
(normalized action-phrase matching, timestamp-aware) as the grouping key -- two messages Part 1 resolved to the same item_id
are by definition the same subject. This keeps Part 1 and Part 2 consistent rather than using two independent, 
possibly contradictory similarity heuristics.

Status is determined by the most recent relevant signal in the group's messages (not majority vote). 
Latest deadline = date in the most recent message that mentions one. Summary is built only from what the messages literally say.
"""

import pandas as pd
import re
import json
import sys
from pathlib import Path


sys.path.insert(0, "scripts")
from priority_engine import (
    build_l1_l2_context, find_linked_item,
    COMPLETED_MARKERS, CANCELLED_MARKERS, RESCHEDULED_MARKERS,
    CONFLICTING_MARKERS,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"

DATE_RE = r"\d{4}-\d{2}-\d{2}"

STATUS_PHRASES = {
    "completed": "The item was ultimately marked completed.",
    "cancelled": "The item was ultimately cancelled.",
    "rescheduled": "The item's schedule was changed.",
    "in_progress": "Status check-ins indicate the item is still in progress.",
    "unclear": "The messages contain conflicting or uncertain status information.",
    "pending": "The item remains pending with no confirmed completion or cancellation.",
}

IN_PROGRESS_MARKERS = re.compile(
    r"\b(is it in progress|in progress|still needs attention|"
    r"has .* been handled|any progress|any update)\b", re.I
)


def determine_group_status(messages_sorted):
    """messages_sorted: [(timestamp, message_id, message_text), ...],
    chronological. The LATEST relevant signal wins."""
    status = "pending"
    for ts, mid, msg in messages_sorted:
        if CANCELLED_MARKERS.search(msg):
            status = "cancelled"
        elif COMPLETED_MARKERS.search(msg):
            status = "completed"
        elif RESCHEDULED_MARKERS.search(msg):
            status = "rescheduled"
        elif CONFLICTING_MARKERS.search(msg):
            status = "unclear"
        elif IN_PROGRESS_MARKERS.search(msg):
            if status not in ("cancelled", "completed"):
                status = "in_progress"
    return status


def determine_latest_deadline(messages_sorted):
    """Date in the CHRONOLOGICALLY LATEST message that has one -- not the
    largest date value, since a later message may move a deadline earlier."""
    latest_date = None
    for ts, mid, msg in messages_sorted:
        m = re.search(DATE_RE, msg)
        if m:
            latest_date = m.group(0)
    return latest_date


def build_summary(messages_sorted, status, phrase):
    n = len(messages_sorted)
    deadline_mentions = sum(1 for _, _, m in messages_sorted if re.search(DATE_RE, m))
    parts = [f"{n} message(s) track '{phrase}'."]
    if deadline_mentions > 1:
        parts.append("The deadline was mentioned or changed more than once across these messages.")
    parts.append(STATUS_PHRASES.get(status, ""))
    return " ".join(p for p in parts if p)


def build_groups(ctx, verbose=True):
    combined = ctx["combined"]
    extract_df = ctx["extract_df"]
    phrase_to_items = ctx["phrase_to_items"]

    message_to_item = {}
    for _, row in combined.iterrows():
        linked = find_linked_item(row["message"], phrase_to_items, at_timestamp=row["timestamp"])
        if linked:
            message_to_item[row["message_id"]] = linked

    item_to_phrase = {}
    for phrase, occurrences in phrase_to_items.items():
        for ts, item_id in occurrences:
            item_to_phrase[item_id] = phrase

    phrase_to_messages = {}
    for mid, item_id in message_to_item.items():
        phrase = item_to_phrase.get(item_id)
        if phrase:
            phrase_to_messages.setdefault(phrase, set()).add(mid)

    extract_lookup = extract_df.set_index("item_id")["source_message_id"].to_dict()
    for phrase, occurrences in phrase_to_items.items():
        for ts, item_id in occurrences:
            origin_mid = extract_lookup.get(item_id)
            if origin_mid:
                phrase_to_messages.setdefault(phrase, set()).add(origin_mid)

    if verbose:
        print(f"Built {len(phrase_to_messages)} candidate groups from phrase linkage")

    combined_indexed = combined.set_index("message_id")
    groups = []
    group_counter = 1

    for phrase, message_ids in phrase_to_messages.items():
        if len(message_ids) < 2:
            continue

        rows = []
        for mid in message_ids:
            if mid in combined_indexed.index:
                row = combined_indexed.loc[mid]
                rows.append((row["timestamp"], mid, row["message"]))
        rows.sort(key=lambda r: r[0])

        status = determine_group_status(rows)
        latest_deadline = determine_latest_deadline(rows)
        summary = build_summary(rows, status, phrase)
        related_item_ids = sorted(set(iid for ts, iid in phrase_to_items.get(phrase, [])))

        base_conf = 0.6 + min(0.03 * len(rows), 0.25)
        if status == "unclear":
            base_conf -= 0.15
        confidence = round(max(0.4, min(0.97, base_conf)), 2)

        group_id = f"GROUP_{group_counter:04d}"
        group_counter += 1

        groups.append({
            "group_id": group_id,
            "title": phrase.capitalize(),
            "related_message_ids": [mid for _, mid, _ in rows],
            "related_item_ids": related_item_ids,
            "status": status,
            "latest_deadline": latest_deadline,
            "summary": summary,
            "confidence": confidence,
        })

    if verbose:
        print(f"Formed {len(groups)} related-message groups (2+ messages each)")

    return groups


if __name__ == "__main__":
    ctx = build_l1_l2_context()
    groups = build_groups(ctx)

    with open(f"{OUT_DIR}/related_message_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    status_counts = pd.Series([g["status"] for g in groups]).value_counts().to_dict()
    print("Status distribution:", json.dumps(status_counts, indent=2))
    print(f"Saved {OUT_DIR}/related_message_groups.json")