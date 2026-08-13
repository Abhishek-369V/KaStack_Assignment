import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MESSAGES_PATH = "./messages.csv"
CLASSIFICATION_PATH = "./classification_results.json"
OUT_DIR = "./outputs"

df = pd.read_csv(MESSAGES_PATH)
with open(CLASSIFICATION_PATH) as f:
    classification_results = json.load(f)

res_df = pd.DataFrame(classification_results)
merged = df.merge(res_df, on="message_id")

# --- Build TF-IDF space over all 900 messages ---
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
tfidf_matrix = vectorizer.fit_transform(merged["message"])

# --- Anchor set: highest-confidence rule-based examples per category ---
HIGH_CONF_THRESHOLD = 0.85
anchors = merged[merged["confidence"] >= HIGH_CONF_THRESHOLD].groupby("category").head(15)
anchor_indices = anchors.index.to_numpy()
anchor_categories = anchors["category"].to_numpy()
anchor_vectors = tfidf_matrix[anchor_indices]

print("Anchor set sizes:")
print(anchors["category"].value_counts())

# --- For every message, find nearest anchor and compare to rule-based label ---
similarities = cosine_similarity(tfidf_matrix, anchor_vectors)  # (900, n_anchors)
nearest_anchor_idx = similarities.argmax(axis=1)
nearest_sim_score = similarities.max(axis=1)
nearest_category = anchor_categories[nearest_anchor_idx]

merged["embedding_nearest_category"] = nearest_category
merged["embedding_similarity"] = nearest_sim_score.round(3)
merged["rule_embedding_agree"] = merged["category"] == merged["embedding_nearest_category"]

agreement_rate = merged["rule_embedding_agree"].mean()
print(f"\nRule-based vs embedding-nearest agreement: {agreement_rate:.1%}")

# --- Save disagreements as a review list ---
disagreements = merged[~merged["rule_embedding_agree"]][
    ["message_id", "message", "category", "confidence",
     "embedding_nearest_category", "embedding_similarity"]
].sort_values("embedding_similarity", ascending=False)

disagreements_out = disagreements.rename(columns={
    "category": "rule_based_category",
}).to_dict(orient="records")

with open(f"{OUT_DIR}/embedding_validation_disagreements.json", "w") as f:
    json.dump(disagreements_out, f, indent=2)

# --- Save full per-message embedding cross-check for transparency ---
full_out = merged[["message_id", "category", "confidence",
                    "embedding_nearest_category", "embedding_similarity",
                    "rule_embedding_agree"]].to_dict(orient="records")
with open(f"{OUT_DIR}/embedding_validation_full.json", "w") as f:
    json.dump(full_out, f, indent=2)

summary = {
    "method": "TF-IDF (1-2 grams) + cosine similarity to per-category anchor examples",
    "anchor_set_size": int(len(anchors)),
    "anchor_source": f"rule-based classifications with confidence >= {HIGH_CONF_THRESHOLD}",
    "total_messages_checked": int(len(merged)),
    "agreement_rate": round(float(agreement_rate), 4),
    "disagreement_count": int(len(disagreements)),
    "note": "general_information has no anchors (rule confidence fixed at 0.60, "
            "below threshold) -- by design, since it's a fallback category with "
            "no distinguishing pattern of its own.",
}
with open(f"{OUT_DIR}/embedding_validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"\nTop 5 disagreements (rule vs embedding-nearest):")
print(disagreements.head(5)[["message_id", "message", "category", "embedding_nearest_category", "embedding_similarity"]].to_string())
