import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Load the saved model components
@st.cache_resource
def load_model_components():
    with open('movie_to_idx.pkl', 'rb') as f:
        movie_to_idx = pickle.load(f)
    with open('idx_to_movie.pkl', 'rb') as f:
        idx_to_movie = pickle.load(f)
    with open('latent_movie_features_train.pkl', 'rb') as f:
        latent_movie_features_train = pickle.load(f)
    with open('k_neighbors_svd_model.pkl', 'rb') as f:
        k_neighbors_svd_model = pickle.load(f)
    return movie_to_idx, idx_to_movie, latent_movie_features_train, k_neighbors_svd_model

movie_to_idx, idx_to_movie, latent_movie_features_train, k_neighbors_svd_model = load_model_components()

def get_recommendation_svd(movie_title, n_recommendations=10):
    # Check if the movie exists in our training dataset's movie titles
    if movie_title not in movie_to_idx:
        return pd.DataFrame(columns=['Movie Title', 'Similarity Score'])

    # Get the index of the target movie
    movie_idx = movie_to_idx[movie_title]

    # Get the latent feature vector for the target movie
    movie_latent_factors = latent_movie_features_train.T[movie_idx].reshape(1, -1)

    # Get distances and indices of n_recommendations + 1 closest movies
    distances, indices = k_neighbors_svd_model.kneighbors(movie_latent_factors, n_neighbors=n_recommendations + 1)

    # Flatten the arrays
    distances = distances.flatten()
    indices = indices.flatten()

    # Create a DataFrame of recommendations
    recommendations_df = pd.DataFrame({
        'Movie Title': [idx_to_movie[i] for i in indices],
        'Similarity Score': 1 - distances  # Convert cosine distance to similarity
    })

    # Filter out the input movie itself and sort by similarity score
    recommendations_df = recommendations_df[recommendations_df['Movie Title'] != movie_title]
    recommendations_df = recommendations_df.sort_values(by='Similarity Score', ascending=False)

    return recommendations_df.head(n_recommendations)

# Streamlit UI
st.title('Movie Recommender System (SVD-based)')
st.write('Select a movie from the dropdown below to get recommendations:')

# Get the list of all available movie titles
all_movie_titles = sorted(list(movie_to_idx.keys()))

# Dropdown for movie selection
selected_movie = st.selectbox('Choose a movie:', all_movie_titles)

# Number of recommendations slider
n_recommendations = st.slider('Number of recommendations:', min_value=5, max_value=20, value=10)

if st.button('Get Recommendations'):
    if selected_movie:
        st.subheader(f'Recommendations for {selected_movie}:')
        recommendations = get_recommendation_svd(selected_movie, n_recommendations)
        if not recommendations.empty:
            st.table(recommendations.reset_index(drop=True))
        else:
            st.write('No recommendations found for this movie.')
    else:
        st.write('Please select a movie.')
