scikit-surprise
matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Set page config
st.set_page_config(page_title="Movie Recommender System", layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    movies = pd.read_csv('data/ml-latest-small/movies.csv')
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    return movies, ratings

# Cache model training
@st.cache_resource
def train_model(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    return model, testset

# Load data
movies, ratings = load_data()

# Train model
model, testset = train_model(ratings)

# Function to get movie recommendations
def get_recommendations(user_id, n=5):
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    all_movies = movies['movieId'].unique()
    movies_to_predict = np.setdiff1d(all_movies, user_movies)
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    recommended_movies = [movies[movies['movieId'] == pred.iid]['title'].iloc[0] for pred in top_n]
    return recommended_movies

# Streamlit UI
st.title('Movie Recommender System using SVD')

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This app uses Singular Value Decomposition (SVD) to recommend movies based on user ratings.")

# Main content
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input('Enter your user ID:', min_value=1, step=1)
    n_recommendations = st.slider('Number of recommendations:', 1, 20, 5)

    if st.button('Get Recommendations'):
        try:
            recommendations = get_recommendations(user_id, n_recommendations)
            st.write('Recommended movies for you:')
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        except ValueError:
            st.error(f"User ID {user_id} not found in the dataset.")

with col2:
    st.subheader('Model Performance')
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")

    st.subheader('Rating Distribution')
    fig, ax = plt.subplots()
    ratings['rating'].hist(bins=10, ax=ax)
    ax.set_title('Distribution of Ratings')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Dataset info
st.subheader('Dataset Information')
n_users = len(ratings.userId.unique())
n_items = len(ratings.movieId.unique())
st.write(f"Number of unique users: {n_users}")
st.write(f"Number of unique movies: {n_items}")
st.write(f"Number of ratings: {len(ratings)}")
sparsity = len(ratings) / (n_users * n_items) * 100
st.write(f"Matrix sparsity: {sparsity:.2f}%")

