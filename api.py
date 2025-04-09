from fastapi import FastAPI
from pydantic import BaseModel
from search import search
import os
import uvicorn

app = FastAPI()

# ✅ Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}

# ✅ Root endpoint (optional)
@app.get("/")
def root():
    return {"message": "FastAPI backend is running"}

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend_assessments(req: QueryRequest):
    try:
        print(f"Received query: {req.query}")

        response = search(
            query=req.query,
            top_k=10,
            debug=False,
            do_rerank=True,
            include_explanations=False
        )

        results = []
        for record in response.get("results", []):
            results.append({
                "url": record.get("URL", ""),
                "adaptive_support": record.get("Adaptive Support", "No"),
                "description": record.get("Description", ""),
                "duration": int(record.get("Duration", 0)),
                "remote_support": record.get("Remote Testing Support", "No"),
                "test_type": record.get("Test Type(s)", [])
            })

        return {
            "recommended_assessments": results
        }
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
