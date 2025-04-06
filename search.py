import faiss
import pickle
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from gemini_booster import rewrite_query, rerank_results, generate_fallback, explain_reasoning

# === Paths ===
INDEX_PATH = "faiss_index.index"
MAPPING_PATH = "index_mapping.pkl"

# === Load Model Safely ===
try:
    model = SentenceTransformer("local_model")
except Exception as e:
    print(f"[ERROR] Failed to load SentenceTransformer model: {e}")
    model = None

# === Cache ===
_index = None
_metadata = None

# === Preprocess Query ===
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Load Index + Metadata ===
def load_index_and_metadata():
    global _index, _metadata

    if _index is None or _metadata is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        if not os.path.exists(MAPPING_PATH):
            raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}")

        _index = faiss.read_index(INDEX_PATH)

        with open(MAPPING_PATH, "rb") as f:
            _metadata = pickle.load(f)

    return _index, _metadata

# === Filters Setup ===
TEST_TYPE_MAP = {
    "ability": "Ability & Aptitude",
    "aptitude": "Ability & Aptitude",
    "biodata": "Biodata & Situational Judgement",
    "situational": "Biodata & Situational Judgement",
    "competency": "Competencies",
    "development": "Development & 360",
    "exercise": "Assessment Exercises",
    "skills": "Knowledge & Skills",
    "knowledge": "Knowledge & Skills",
    "personality": "Personality & Behavior",
    "behavior": "Personality & Behavior",
    "simulation": "Simulations"
}
DURATION_REGEX = re.compile(r"(\d+)\s*minutes?")
JOB_LEVELS = ["entry", "mid", "senior", "executive", "graduate", "manager"]

def extract_filters_from_prompt(prompt):
    prompt = prompt.lower()
    filters = {}

    if match := DURATION_REGEX.search(prompt):
        filters["max_duration"] = int(match.group(1))

    filters["job_levels"] = [level for level in JOB_LEVELS if level in prompt]

    filters["test_types"] = [
        decoded for key, decoded in TEST_TYPE_MAP.items() if key in prompt
    ]

    return filters

def passes_filters(record, filters):
    try:
        duration_str = str(record.get("Duration", "")).split()
        duration_val = int(duration_str[0]) if duration_str else None
    except:
        duration_val = None

    if "max_duration" in filters and duration_val and duration_val > filters["max_duration"]:
        return False

    if filters.get("job_levels"):
        jl = str(record.get("Job Levels", "")).lower()
        if not any(level in jl for level in filters["job_levels"]):
            return False

    if filters.get("test_types"):
        tt = str(record.get("Decoded Test Type(s)", "")).lower()
        if not any(t.lower() in tt for t in filters["test_types"]):
            return False

    return True

# === Main Search ===
def search(query, top_k=10, debug=False, include_explanations=False, do_rerank=True):
    try:
        index, metadata = load_index_and_metadata()
    except Exception as e:
        print(f"[ERROR] Index/Metadata load failed: {e}")
        return {
            "rewritten_query": query,
            "results": [],
            "fallback": f"Could not load index or metadata: {e}"
        }

    if not model:
        return {
            "rewritten_query": query,
            "results": [],
            "fallback": "SentenceTransformer model not loaded."
        }

    rewritten_query = rewrite_query(query)
    if debug:
        print(f"📝 Rewritten Query: {rewritten_query}")

    filters = extract_filters_from_prompt(rewritten_query)
    if debug:
        print("🔍 Extracted Filters:", filters)

    try:
        query_embedding = model.encode([preprocess(rewritten_query)], show_progress_bar=False)
        query_embedding = normalize(query_embedding, axis=1)
        distances, indices = index.search(query_embedding, top_k * 5)
    except Exception as e:
        print(f"[ERROR] FAISS search failed: {e}")
        return {
            "rewritten_query": rewritten_query,
            "results": [],
            "fallback": generate_fallback(query)
        }

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx >= len(metadata):
            continue
        record = metadata[idx].copy()
        record["Score"] = float(score)

        if passes_filters(record, filters):
            record.pop("Decoded Test Type(s)", None)
            results.append(record)

        if len(results) >= top_k:
            break

    if not results:
        return {
            "rewritten_query": rewritten_query,
            "results": [],
            "fallback": generate_fallback(query)
        }

    if do_rerank:
        try:
            results = rerank_results(rewritten_query, results)
        except Exception as e:
            print(f"[WARN] Rerank failed: {e}")

    if include_explanations:
        for r in results:
            try:
                r["LLM Explanation"] = explain_reasoning(rewritten_query, r)
            except Exception as e:
                r["LLM Explanation"] = f"Error generating explanation: {e}"

    return {
        "rewritten_query": rewritten_query,
        "results": results[:top_k]
    }

# === CLI Debug Mode ===
if __name__ == "__main__":
    q = input("Enter your hiring need or requirement: ")
    res = search(q, top_k=5, debug=True, include_explanations=True)
    print("\n🔎 Results:")
    if res.get("results"):
        for i, r in enumerate(res["results"], 1):
            print(f"\n{i}. {r['Assessment Name']} | {r['Duration']} | {r['Job Levels']}")
            print(f"   🔍 Explanation: {r.get('LLM Explanation', 'N/A')}")
    else:
        print(f"💬 Gemini Fallback: {res.get('fallback')}")
