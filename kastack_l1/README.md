# Message Intelligence Pipeline

A rule-based system that classifies 900 messages into 6 categories, extracts
tasks/events, and detects + masks sensitive information.

**Approach**: A rule-based (regex/pattern) engine is used as the primary classifier because the dataset is templated and a transparent, fully-explainable engine is more auditable and accurate here than a black-box model trained on 900 rows. A TF-IDF + cosine-similarity embedding layer is added afterward as an independent cross-check on the rule-based results (see Part 4).

The notebook(message_intelligence_pipeline) is organised to match the assignment's structure exactly:
- **Part 1** — Message Classification
- **Part 2** — Task & Event Extraction
- **Part 3** — Sensitive Information Detection
- **Part 4** — Embedding-based validation (ML component) - Added by me
- **Part 5** — Mandatory demo IDs + summary stats - Added by me

## Why rule-based, not ML

The dataset is templated (fixed sentence patterns with variable dates/names/
values). A trained classifier on 900 templated rows would overfit to the
templates and be a black box I'd struggle to fully justify line-by-line in
the video. A transparent regex/pattern engine means every single decision
traces back to one named rule — which directly satisfies the "must explain
every decision" and "must understand everything submitted" requirements.
Traditional ML/embeddings were considered but rejected for this reason, not
because of unfamiliarity with them.

## How classification works (Part 1)

Messages are checked against ordered pattern groups. First match wins.
Priority order matters because messages can superficially match more than
one category:

1. **Sensitive information** — checked first, always wins (safety-first)
2. **Promotional** — SAVE codes, discount/sale language
3. **Meeting or event** — "Calendar update", "happens on <date>", "scheduled for"
4. **Action required** — explicit deadlines, "by/before <date>" + action verb,
   direct requests ("Can you...", "I need you to...", "Don't forget to...")
5. **Personal information** — "Personal note:", first-person disclosures
   ("my emergency contact is...", "i prefer...")
6. **General information** — default fallback when nothing else matches

Each result includes `message_id`, `category`, `confidence` (fixed per rule
tier, not learned), and a `reason` string naming the specific rule matched.

## How task/event extraction works (Part 2)

Runs only on messages already classified `action_required` or
`meeting_or_event`. For each:
- **date/time**: extracted via regex (`YYYY-MM-DD`, `HH:MM`, or `H AM/PM`)
- **title**: message with boilerplate prefixes ("FYI:", "Quick update:", etc.)
  and trailing date clauses stripped
- **person**: only filled if the message explicitly names someone via a
  "with <Name>" pattern; otherwise `"unresolved"`
- **priority**: `high` if a concrete date/deadline exists, else `medium`

**Nothing is guessed.** Any field the regex can't confidently resolve is
stored as the literal string `"unresolved"` — never inferred or fabricated.

## How sensitive detection works (Part 3)

A separate pattern table matches known sensitive phrasings (password, OTP,
account recovery code, card number, auth token, home address, ID number,
phone number, bank account number). On match:
- the raw value is masked (`first 2 chars + asterisks + last 2 chars`)
- risk is set (`high` for credentials/financial data, `medium` for address/phone)
- `recommended_action` is `do_not_store` for high risk, `ask_for_confirmation`
  for medium risk

This check runs **before** classification, so a sensitive message is never
misfiled into another category and its raw value is masked before appearing
in any downstream output, log, or (for the video) on-screen display.

## Assumptions

- Timestamps in the CSV are already trustworthy and chronological once sorted
- "Mandatory" priority for tasks = presence of a concrete date; there's no
  urgency/importance signal in the text beyond that
- A message can only belong to one category (no multi-label)

## Limitations

- Regex over generic verbs ("please", "send") originally over-triggered
  `action_required` on unrelated filler sentences (e.g. "Please note: Flash
  sale...") — fixed by requiring either a date constraint or a specific
  high-confidence phrase, and by checking `promotional`/`meeting` before
  `action_required`.
- Some messages are genuinely ambiguous even to a human, e.g. "The review
  could be Friday afternoon" (no explicit request — classified
  `general_information`, arguably could be `meeting_or_event`) and
  "Maya asked whether the demo was ready" (a question *about* a task, not
  a task assigned to the recipient — classified `general_information`).
- `person` extraction is intentionally conservative and returns `unresolved`
  for the vast majority of items, since the dataset rarely names a second
  party explicitly.


## AI-tool usage declaration
- Taken help of Claude (Anthropic) for: 
exploring the dataset structure, iterating on regex patterns after reviewing false positives, drafting this README, and assisting in styling_frontend(streamlit).

## Declaration:
All logic was reviewed, tested against sample output, and is understood and explainable by me. No external API was used to classify or process the actual message content — everything runs locally via Python/pandas/regex, satisfying the "do not send raw messages to external AI services" rule.

## Output files

- `classification_results.json` — all 900 messages, category + confidence + reason
- `extraction_results.json` — all extracted tasks/events
- `sensitive_detection_results.json` — all flagged sensitive messages, masked
- `mandatory_ids_results.json` — classification results for the 15 required IDs
- `summary.json` — category counts and totals
