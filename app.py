import os
from flask import Flask, render_template, request, jsonify
from recommender import get_recommender

app = Flask(__name__)
engine = get_recommender()

@app.route("/", methods=["GET", "POST"])
def home():
    """Serves the web UI template with form handling and autocomplete choices."""
    recommendations = []
    selected_movie = ""
    mode = "content"

    if request.method == "POST":
        selected_movie = request.form.get("movie", "")
        mode = request.form.get("mode", "content")
        if selected_movie:
            if mode == "collaborative":
                try:
                    user_id = int(selected_movie)
                    recommendations = engine.recommend_collaborative(user_id=user_id, top_n=5)
                except ValueError:
                    recommendations = engine.recommend_content(selected_movie, top_n=5)
            else:
                recommendations = engine.recommend_content(selected_movie, top_n=5)

    movies = engine.get_movie_titles()

    return render_template(
        "index.html",
        movies=movies,
        recommendations=recommendations,
        selected_movie=selected_movie,
        mode=mode
    ), 200

@app.route("/recommend/content", methods=["GET"])
def recommend_content_api():
    """API endpoint for content-based TF-IDF movie similarity search."""
    try:
        title = request.args.get("title", "Toy Story (1995)")
        top_n = int(request.args.get("top_n", 5))
        results = engine.recommend_content(title, top_n=top_n)
        return jsonify({"query_title": title, "recommendations": results}), 200
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400

@app.route("/recommend/collaborative", methods=["GET"])
def recommend_collaborative_api():
    """API endpoint for collaborative SVD matrix factorization recommendations."""
    try:
        user_id = int(request.args.get("user_id", 1))
        top_n = int(request.args.get("top_n", 5))
        results = engine.recommend_collaborative(user_id=user_id, top_n=top_n)
        return jsonify({"user_id": user_id, "recommendations": results}), 200
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)