import pandas as pd
import re
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import os

# File paths
CSV_FILES = ["shl_data_type1.csv"]
INDEX_PATH = "faiss_index.index"
MAPPING_PATH = "index_mapping.pkl"

# Load model
print("📥 Loading model: all-MiniLM-L6-v2")
model = SentenceTransformer("local_model")

# Preprocessing utility (for embeddings only)
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Human-readable decoding
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

def decode_test_types(test_type_str):
    return [TEST_TYPE_MAP.get(t.strip(), t.strip()) for t in test_type_str.split(',') if t.strip()]

# Load CSV, prepare text and metadata
def load_and_prepare_data():
    all_texts = []
    metadata = []

    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            print(f"⚠️ File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file).fillna("")

        for _, row in df.iterrows():
            decoded_test_types = decode_test_types(row.get("Test Type(s)", ""))
            decoded_test_types_str = ", ".join(decoded_test_types)

            # Combined text for semantic embedding
            combined_text = " | ".join([
                row.get("Assessment Name", ""),
                decoded_test_types_str,
                row.get("Job Levels", ""),
                row.get("Description", "")
            ])
            all_texts.append(preprocess(combined_text))

            # Raw metadata for Gemini reranking/explanation
            metadata.append({
                "Assessment Name": row.get("Assessment Name", ""),
                "URL": row.get("URL", ""),
                "Remote Testing Support": row.get("Remote Testing Support", ""),
                "Adaptive Support": row.get("Adaptive Support", ""),
                "IRT Support": row.get("IRT Support", ""),
                "Duration": row.get("Duration", ""),
                "Test Type(s)": row.get("Test Type(s)", ""),             # Original format
                "Decoded Test Type(s)": decoded_test_types_str,          # ✅ Add this line
                "Job Levels": row.get("Job Levels", ""),
                "Languages": row.get("Languages", ""),
                "Description": row.get("Description", "")
            })


    return all_texts, metadata

# FAISS index creation
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Main pipeline
def main():
    print("🔄 Loading and preparing data...")
    texts, metadata = load_and_prepare_data()

    if not texts:
        print("❌ No data found to embed. Exiting.")
        return

    print(f"🧠 Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings, axis=1)

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"💾 Saving index to: {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"💾 Saving metadata mapping to: {MAPPING_PATH}")
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Embedding and indexing complete.")

if __name__ == "__main__":
    main()
