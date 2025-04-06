from flask import Flask, request, jsonify
from search import search
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ SHL Assessment Recommendation API (Flask) is running!"

@app.route("/recommend", methods=["POST"])
def recommend_assessments():
    try:
        data = request.get_json()
        query = data.get("query")
        top_k = int(data.get("top_k", 10))
        rerank = data.get("rerank", True)
        fallback = data.get("fallback", True)
        explanations = data.get("explanations", False)

        print(f"Received query: {query}")

        response = search(
            query=query,
            top_k=top_k,
            debug=False,
            do_rerank=rerank,
            include_explanations=explanations
        )

        print("Search complete")

        return jsonify({
            "status": "success",
            "rewritten_query": response.get("rewritten_query", ""),
            "results": response.get("results", []),
            "fallback": response.get("fallback", None)
        })

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"status": "error", "message": str(e)})

# For Render dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
