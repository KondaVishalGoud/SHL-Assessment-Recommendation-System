from flask import Flask, request, jsonify, render_template_string
from search import search

app = Flask(__name__)

# Simple HTML form template
FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Assessment Recommendation</title></head>
<body>
  <h2>Enter your hiring need or query:</h2>
  <form method="post">
    <input type="text" name="query" placeholder="e.g. Java developer, 30 mins, entry level" size="50" required>
    <br><br>
    <input type="submit" value="Get Recommendations">
  </form>
</body>
</html>
"""

@app.route("/")
def home():
    return "✅ Flask API is running!"

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "GET":
        return render_template_string(FORM_TEMPLATE)
    
    if request.method == "POST":
        query = request.form.get("query") or request.json.get("query")
        top_k = int(request.form.get("top_k", 10)) if request.form else 10
        rerank = True
        explanations = False

        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400

        try:
            response = search(
                query=query,
                top_k=top_k,
                debug=False,
                do_rerank=rerank,
                include_explanations=explanations
            )
            return jsonify({
                "status": "success",
                "rewritten_query": response.get("rewritten_query", ""),
                "results": response.get("results", []),
                "fallback": response.get("fallback", None)
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
