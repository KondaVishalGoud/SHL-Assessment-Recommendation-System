# api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from search import search

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank: bool = True
    fallback: bool = True
    explanations: bool = False

@app.post("/recommend")
def recommend_assessments(req: QueryRequest):
    try:
        print(f"Received query: {req.query}")  # Add this line

        response = search(
            query=req.query,
            top_k=req.top_k,
            debug=False,
            do_rerank=req.rerank,
            include_explanations=req.explanations
        )

        print("Search complete")  # Add this line

        return {
            "status": "success",
            "rewritten_query": response.get("rewritten_query", ""),
            "results": response.get("results", []),
            "fallback": response.get("fallback", None)
        }
    except Exception as e:
        print(f"Error occurred: {e}")  # Add this line
        return {"status": "error", "message": str(e)}

