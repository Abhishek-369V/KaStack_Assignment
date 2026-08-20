"""
Message Intelligence Pipeline (L1, refactored for L2 reuse)
Parts 1-3: Classification, Task/Event Extraction, Sensitive Info Detection
"""

import pandas as pd
import re
import json
import os
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "messages.csv"
MANDATORY_PATH = BASE_DIR/ "mandatory_demo_ids.csv"
OUT_DIR = BASE_DIR / "outputs"

DATE_RE = r"\d{4}-\d{2}-\d{2}"
TIME_RE = r"\d{1,2}(:\d{2})?\s?(AM|PM)?"

# ---------------------------------------------------------------------
# PART 3: Sensitive Info Detection
# ---------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    ("password", re.compile(r"password\s+([A-Za-z0-9#\-]+)", re.I), "high"),
    ("one_time_password", re.compile(r"OTP\s+is\s+([0-9\-]+)", re.I), "high"),
    ("account_recovery_code", re.compile(r"recovery code is\s+([A-Za-z0-9\-]+)", re.I), "high"),
    ("card_number", re.compile(r"card number is\s+([0-9 \-]+)", re.I), "high"),
    ("auth_token", re.compile(r"access token is\s+([A-Za-z0-9_\-]+)", re.I), "high"),
    ("home_address", re.compile(r"home address is\s+(.+?)(?:\.|$)", re.I), "medium"),
    ("identification_number", re.compile(r"identification number is\s+([A-Za-z0-9\-]+)", re.I), "high"),
    ("phone_number", re.compile(r"contact me on\s+([\d \-]+)", re.I), "medium"),
    ("bank_account_number", re.compile(r"bank account number\s+([\d\-]+)", re.I), "high"),
]

def mask_value(val):
    val = val.strip().rstrip(".")
    if len(val) <= 4:
        return "*" * len(val)
    return val[:2] + "*" * (len(val) - 4) + val[-2:]

def detect_sensitive(msg):
    for stype, pattern, risk in SENSITIVE_PATTERNS:
        m = pattern.search(msg)
        if m:
            raw = m.group(1)
            masked_val = mask_value(raw)
            masked_text = msg[:m.start(1)] + masked_val + msg[m.end(1):]
            action = "do_not_store" if risk == "high" else "ask_for_confirmation"
            return {
                "sensitivity_type": stype,
                "risk": risk,
                "masked_text": masked_text,
                "recommended_action": action,
            }
    return None

# ---------------------------------------------------------------------
# PART 1: Classification
# ---------------------------------------------------------------------

PROMO_PATTERNS = [
    (re.compile(r"\bUse code SAVE\d+\b", re.I), "promo code present"),
    (re.compile(r"\b(discount|off selected|flash sale|subscription|premium plan|reward points|coupon|save \d+%)\b", re.I),
     "promotional/marketing language"),
]

MEETING_PATTERNS = [
    (re.compile(r"\bCalendar update\b", re.I), "explicit calendar update phrasing"),
    (re.compile(r"\bhappens on\s+" + DATE_RE, re.I), "'happens on <date>' meeting/reminder phrasing"),
    (re.compile(r"\bscheduled for\s+" + DATE_RE, re.I), "'scheduled for <date>' phrasing"),
    (re.compile(r"\bjoin the .* on\s+" + DATE_RE, re.I), "invite to join an event on a date"),
    (re.compile(r"\bavailable for the .* at\b", re.I), "availability request for an event"),
]

ACTION_PATTERNS = [
    (re.compile(r"\bdeadline is\s+" + DATE_RE, re.I), "explicit deadline stated"),
    (re.compile(r"\bis due on\s+" + DATE_RE, re.I), "explicit due date stated"),
    (re.compile(r"\b(review|reply|confirm|complete|upload|submit|renew|pay|email|update|book|send|finish)\b.*\b(by|before)\s+" + DATE_RE, re.I),
     "action verb requested with a date constraint ('by'/'before')"),
    (re.compile(r"^(Can you|Could you) (review|update|confirm|send|call|finish)\b", re.I), "direct question requesting an action"),
    (re.compile(r"\bI need you to\b", re.I), "explicit request phrasing"),
    (re.compile(r"\bDon'?t forget to\b", re.I), "explicit reminder-to-act phrasing"),
    (re.compile(r"\bPlease call\b", re.I), "explicit request to call someone"),
]

PERSONAL_PATTERNS = [
    (re.compile(r"\bPersonal note\b", re.I), "explicit 'Personal note' tag"),
    (re.compile(r"\bmy (favourite|emergency contact|home address|identification number)\b", re.I),
     "first-person personal detail disclosed"),
    (re.compile(r"\bi (am|drink|use|prefer)\b", re.I), "first-person preference/trait statement"),
    (re.compile(r"\btest result\b", re.I), "personal health information"),
]

def classify(msg):
    sens = detect_sensitive(msg)
    if sens:
        stype = sens["sensitivity_type"]
        return "sensitive_information", 0.97, f"Matches {stype} pattern; flagged before general classification for safety.", sens

    for pattern, reason in PROMO_PATTERNS:
        if pattern.search(msg):
            return "promotional", 0.90, reason, None

    for pattern, reason in MEETING_PATTERNS:
        if pattern.search(msg):
            return "meeting_or_event", 0.88, reason, None

    for pattern, reason in ACTION_PATTERNS:
        if pattern.search(msg):
            return "action_required", 0.87, reason, None

    for pattern, reason in PERSONAL_PATTERNS:
        if pattern.search(msg):
            return "personal_information", 0.85, reason, None

    return "general_information", 0.60, "No task, event, personal, promotional, or sensitive pattern matched; treated as informational default.", None

# ---------------------------------------------------------------------
# PART 2: Task / Event Extraction
# ---------------------------------------------------------------------

def extract_date(msg):
    m = re.search(DATE_RE, msg)
    return m.group(0) if m else None

def extract_time(msg):
    m = re.search(r"\b(\d{1,2}:\d{2})\b", msg)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{1,2}\s?(AM|PM))\b", msg, re.I)
    if m2:
        return m2.group(1)
    return None

def extract_person(msg, sender):
    m = re.search(r"\bwith\s+([A-Z][a-z]+)\b", msg)
    if m:
        return m.group(1)
    return None

def extract_title(msg):
    cleaned = re.sub(r"^(Hi,|FYI:|Please note:|Just checking—|Can you help\?|"
                      r"For today:|One more thing:|Quick update:|Important:)\s*",
                      "", msg).strip()
    cleaned = re.split(r"\s*(?:before|by|deadline is|is due on|happens on|scheduled for)\b",
                        cleaned, maxsplit=1)[0].strip(" .,;:?")
    cleaned = re.sub(r"^(Can you|Could you|Don'?t forget to|I need you to|Please)\s+", "", cleaned, flags=re.I)
    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
    return cleaned if cleaned else msg.strip()

def extract_task_or_event(row, category):
    msg = row["message"]
    date = extract_date(msg)
    time_ = extract_time(msg)
    person = extract_person(msg, row["sender"])
    title = extract_title(msg)

    if category == "action_required":
        item_type = "task"
        priority = "high" if date else "medium"
    else:
        item_type = "event"
        priority = "medium"

    return {
        "type": item_type,
        "title": title if title else "unresolved",
        "description": msg,
        "date_or_deadline": date if date else "unresolved",
        "time": time_ if time_ else "unresolved",
        "person": person if person else "unresolved",
        "priority": priority,
        "source_message_id": row["message_id"],
    }

# ---------------------------------------------------------------------
# Runner: importable, no side effects unless called explicitly
# ---------------------------------------------------------------------

def run_l1_pipeline(in_path=IN_PATH, mandatory_path=MANDATORY_PATH, out_dir=OUT_DIR, save=True):
    """Runs classification, extraction, sensitive detection. Returns the
    three result lists. Only executes when explicitly called -- importing
    this module has no side effects."""
    df = pd.read_csv(in_path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    mandatory_ids = pd.read_csv(mandatory_path)["message_id"].tolist()

    classification_results = []
    extraction_results = []
    sensitive_results = []

    task_counter = 1
    for _, row in df.iterrows():
        msg = row["message"]
        category, confidence, reason, sens_info = classify(msg)

        classification_results.append({
            "message_id": row["message_id"],
            "category": category,
            "confidence": confidence,
            "reason": reason,
        })

        if sens_info:
            sensitive_results.append({
                "message_id": row["message_id"],
                **sens_info,
            })

        if category in ("action_required", "meeting_or_event"):
            item = extract_task_or_event(row, category)
            item["item_id"] = f"{'TASK' if item['type']=='task' else 'EVENT'}_{task_counter:04d}"
            task_counter += 1
            extraction_results.append(item)

    if save:
        os.makedirs(out_dir, exist_ok=True)

        with open(f"{out_dir}/classification_results.json", "w") as f:
            json.dump(classification_results, f, indent=2)
        with open(f"{out_dir}/extraction_results.json", "w") as f:
            json.dump(extraction_results, f, indent=2)
        with open(f"{out_dir}/sensitive_detection_results.json", "w") as f:
            json.dump(sensitive_results, f, indent=2)

        mandatory_subset = [c for c in classification_results if c["message_id"] in mandatory_ids]
        with open(f"{out_dir}/mandatory_ids_results.json", "w") as f:
            json.dump(mandatory_subset, f, indent=2)

        cat_counts = pd.Series([c["category"] for c in classification_results]).value_counts().to_dict()
        summary = {
            "total_messages": len(df),
            "category_counts": cat_counts,
            "total_tasks_extracted": sum(1 for e in extraction_results if e["type"] == "task"),
            "total_events_extracted": sum(1 for e in extraction_results if e["type"] == "event"),
            "total_sensitive_flagged": len(sensitive_results),
            "mandatory_ids_covered": len(mandatory_subset),
        }
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

    return classification_results, extraction_results, sensitive_results


if __name__ == "__main__":
    run_l1_pipeline()