import os
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

# 🔐 Load environment variable from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Make sure it's defined in your .env file.")

# 🔑 Configure Gemini
genai.configure(api_key=api_key)

# Load Gemini Pro model
model = genai.GenerativeModel("gemini-1.5-flash")

# 🌀 Rewrite Query
def rewrite_query(original_query: str) -> str:
    prompt = f"""You are a helpful assistant. Rewrite this vague or ambiguous hiring query into a more specific and structured version suited for matching with assessment tests.

Original Query:
{original_query}

Rewritten Query:"""
    try:
        response = model.generate_content(prompt)
        rewritten = response.text.strip()
        print(f"\n🔁 Gemini Rewritten Query:\n{rewritten}\n")
        return rewritten
    except Exception as e:
        print(f"⚠️ Rewrite failed: {e}")
        return original_query


# 🔁 Rerank Results
def rerank_results(query: str, results: List[Dict]) -> List[Dict]:
    if not results:
        return []

    context = "\n".join([
        f"{i+1}. {r['Assessment Name']} - {r.get('Description', '')[:200]}..." for i, r in enumerate(results)
    ])
    prompt = f"""You are an AI that reranks test assessments based on how relevant they are to a given hiring query.

Query: {query}

Here are the current top results:
{context}

Return a new ranking as a list of assessment names from most to least relevant.
Output Format (just list): 
1. Assessment A
2. Assessment B
...

Reranked List:"""

    response = model.generate_content(prompt)
    names = [line.split(". ", 1)[-1].strip() for line in response.text.strip().splitlines() if ". " in line]
    name_to_result = {r["Assessment Name"]: r for r in results}

    reranked = []
    for name in names:
        if name in name_to_result:
            reranked.append(name_to_result[name])
        else:
            print(f"[⚠️ Warning] Gemini returned unknown name: {name}")

    return reranked if reranked else results  # Fallback to original if rerank fails


# 💡 Fallback Generation
def generate_fallback(query: str) -> str:
    prompt = f"""No relevant assessments were found for the query below. Provide a helpful message or alternative suggestion.

Query: {query}

Response:"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Sorry, no matching assessments were found. Please try rephrasing your input."

# 🧠 Explain Recommendation
def explain_reasoning(query: str, assessment: Dict) -> str:
    prompt = f"""Explain in 2-3 lines why the following assessment is suitable for this hiring query.

Query: {query}

Assessment:
Name: {assessment['Assessment Name']}
Description: {assessment.get('Description', '')}
Test Type: {assessment.get('Test Type(s)', '')}
Job Levels: {assessment.get('Job Levels', '')}

Explanation:"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "This assessment aligns well with the job requirements based on type, level, and content."
