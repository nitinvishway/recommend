import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class HybridMovieRecommender:
    """
    Hybrid Movie Recommendation Engine combining:
    1. Content-Based Filtering via TF-IDF Vectorization & Cosine Similarity.
    2. Collaborative Filtering via TruncatedSVD Matrix Factorization.
    """
    def __init__(self, movies_path="movies.csv", ratings_path="ratings.csv"):
        print("Initializing Hybrid Movie Recommender Engine...")
        self.movies = pd.read_csv(movies_path)
        self.movies["genres"] = self.movies["genres"].fillna("").str.replace("|", " ")
        self.movies["title_clean"] = self.movies["title"].fillna("")
        
        # 1. Content-Based TF-IDF Matrix
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["genres"] + " " + self.movies["title_clean"])
        self.title_to_idx = pd.Series(self.movies.index, index=self.movies["title"]).drop_duplicates()

        # 2. Collaborative Filtering Matrix Factorization (if ratings available)
        self.has_ratings = False
        try:
            self.ratings = pd.read_csv(ratings_path)
            # Create User-Item Matrix
            user_item_matrix = self.ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
            self.user_ids = list(user_item_matrix.index)
            self.movie_ids = list(user_item_matrix.columns)
            
            # TruncatedSVD Matrix Factorization
            n_components = min(20, user_item_matrix.shape[1] - 1)
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.user_factors = self.svd.fit_transform(user_item_matrix)
            self.item_factors = self.svd.components_
            
            self.user_item_df = user_item_matrix
            self.has_ratings = True
            print("Collaborative Filtering Matrix Factorization initialized successfully.")
        except Exception as e:
            print(f"Notice: Ratings matrix initialization fallback: {e}")

    def recommend_content(self, movie_title, top_n=5):
        """Content-based recommendations using TF-IDF cosine similarity."""
        if movie_title not in self.title_to_idx:
            # Fuzzy match if exact title not found
            matches = self.movies[self.movies["title"].str.contains(movie_title, case=False, na=False)]
            if len(matches) > 0:
                movie_title = matches.iloc[0]["title"]
            else:
                return []

        idx = self.title_to_idx[movie_title]
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]

        results = []
        for i in top_indices:
            row = self.movies.iloc[i]
            results.append({
                "title": row["title"],
                "genres": row["genres"].replace(" ", " | "),
                "similarity_score": f"{float(sim_scores[i] * 100):.1f}%"
            })
        return results

    def recommend_collaborative(self, user_id=1, top_n=5):
        """Collaborative recommendations using TruncatedSVD Matrix Factorization."""
        if not self.has_ratings or user_id not in self.user_item_df.index:
            user_id = self.user_ids[0] if self.has_ratings else 1

        if self.has_ratings:
            user_idx = self.user_item_df.index.get_loc(user_id)
            user_pred_ratings = np.dot(self.user_factors[user_idx, :], self.item_factors)
            
            # Mask movies already rated by user
            already_rated = self.user_item_df.loc[user_id] > 0
            user_pred_ratings[already_rated] = -1
            
            top_movie_indices = user_pred_ratings.argsort()[-top_n:][::-1]
            results = []
            for idx in top_movie_indices:
                movieId = self.movie_ids[idx]
                movie_row = self.movies[self.movies["movieId"] == movieId]
                if len(movie_row) > 0:
                    row = movie_row.iloc[0]
                    score = min(5.0, max(1.0, float(user_pred_ratings[idx])))
                    results.append({
                        "title": row["title"],
                        "genres": row["genres"].replace(" ", " | "),
                        "predicted_rating": f"★ {score:.1f} / 5.0"
                    })
            return results
        return self.recommend_content("Toy Story (1995)", top_n=top_n)

    def get_movie_titles(self):
        """Returns sorted list of all available movie titles."""
        return sorted(self.movies["title"].tolist())

# Instantiate engine singleton
_recommender_instance = None

def get_recommender():
    global _recommender_instance
    if _recommender_instance is None:
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        movies_p = os.path.join(base_dir, "movies.csv")
        ratings_p = os.path.join(base_dir, "ratings.csv")
        _recommender_instance = HybridMovieRecommender(movies_p, ratings_p)
    return _recommender_instance

def recommend(movie_title, top_n=5):
    engine = get_recommender()
    recs = engine.recommend_content(movie_title, top_n=top_n)
    return [r["title"] for r in recs]

def get_movie_titles():
    engine = get_recommender()
    return engine.get_movie_titles()