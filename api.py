# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from search import search
import os
import uvicorn

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
        print(f"Received query: {req.query}")

        response = search(
            query=req.query,
            top_k=req.top_k,
            debug=False,
            do_rerank=req.rerank,
            include_explanations=req.explanations
        )

        print("Search complete")

        return {
            "status": "success",
            "rewritten_query": response.get("rewritten_query", ""),
            "results": response.get("results", []),
            "fallback": response.get("fallback", None)
        }
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"status": "error", "message": str(e)}

# Required for Render.com (detects PORT and binds correctly)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
