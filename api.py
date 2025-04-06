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
        response = search(
            query=req.query,
            top_k=req.top_k,
            debug=False,
            do_rerank=req.rerank,
            include_explanations=req.explanations
        )
        return {
            "status": "success",
            "rewritten_query": response.get("rewritten_query", ""),
            "results": response.get("results", []),
            "fallback": response.get("fallback", None)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
