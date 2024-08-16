import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore

# 1. Load and Prepare Data
# Example dataset containing movie titles, genres, and user ratings.
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II'],
    'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy']
})

ratings = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'movie_id': [1, 2, 3, 4, 5, 2, 3, 4, 5, 1],
    'rating': [5, 4, 3, 4, 5, 3, 4, 2, 5, 4]
})

# 2. Content-Based Filtering
def content_based_recommender(movie_title, movies, n=3):
    # Calculate TF-IDF for genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the movie that matches the title
    idx = movies.index[movies['title'] == movie_title].tolist()[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the n most similar movies
    sim_scores = sim_scores[1:n+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top n most similar movies
    return movies['title'].iloc[movie_indices]

# Example usage of content-based filtering
print("Content-Based Recommendations for 'Toy Story':")
print(content_based_recommender('Toy Story', movies))

# 3. Collaborative Filtering
def collaborative_filtering_recommender(user_id, ratings, movies, n=3):
    # Create a user-item matrix
    user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
    
    # Fill NaN values with 0 (assumes no rating means no preference)
    user_movie_matrix.fillna(0, inplace=True)
    
    # Compute cosine similarity between users
    user_sim = cosine_similarity(user_movie_matrix)
    
    # Find the nearest neighbors for the target user
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_matrix)
    
    distances, indices = knn.kneighbors([user_movie_matrix.loc[user_id]], n_neighbors=n+1)
    
    # Get the neighbors' indices
    neighbors_indices = indices.flatten()[1:]
    
    # Recommend movies that the neighbors have liked but the user hasn't seen yet
    neighbor_ratings = user_movie_matrix.iloc[neighbors_indices]
    neighbor_ratings = neighbor_ratings.mean(axis=0)
    unseen_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] == 0].index.tolist()
    
    recommendations = neighbor_ratings.loc[unseen_movies].sort_values(ascending=False).head(n)
    
    # Handle case where movie_id is not in the original movies DataFrame
    valid_recommendations = recommendations.index[recommendations.index.isin(movies['movie_id'])]
    
    if valid_recommendations.empty:
        return "No recommendations available."
    
    return movies.loc[movies['movie_id'].isin(valid_recommendations), 'title']

# Example usage of collaborative filtering
print("\nCollaborative Filtering Recommendations for User 1:")
print(collaborative_filtering_recommender(1, ratings, movies))
