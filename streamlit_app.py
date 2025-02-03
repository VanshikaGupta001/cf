import subprocess

# Run the curl command
subprocess.run(["curl", "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "-o", "ml-latest-small.zip"])


import zipfile
with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# load data
movies = pd.read_csv('data/ml-latest-small/movies.csv')
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

# check for missing values 
movie_names = movies.set_index('movieId')['title'].to_dict()
n_users = len(ratings.userId.unique())
n_items = len(ratings.movieId.unique())
print("Number of unique users:", n_users)
print("Number of unique movies:", n_items)
print("The full rating matrix will have:", n_users*n_items, 'elements.')
print('----------')
print("Number of ratings:", len(ratings))
print("Therefore: only", len(ratings) / (n_users*n_items) * 100, '% of the matrix is filled.')
print("incredibly sparse matrix to work with here.")

# Create Surprise dataset
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Train the SVD model
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# Function to get movie recommendations
def get_recommendations(user_id, n=5):
    # Get all movies the user hasn't rated
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    all_movies = movies['movieId'].unique()
    movies_to_predict = np.setdiff1d(all_movies, user_movies)
    
    # Predict ratings for all movies the user hasn't seen
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]
    
    # Sort predictions by estimated rating
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # Get movie titles
    recommended_movies = [movies[movies['movieId'] == pred.iid]['title'].iloc[0] for pred in top_n]
    
    return recommended_movies



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load data and train model (as shown in the previous code snippet)

st.title('Movie Recommender System using SVD')

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

# Add a section to show model performance
if st.checkbox('Show Model Performance'):
    # Evaluate the model on the test set
    from surprise import accuracy
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")

# Visualize rating distribution
plt.figure(figsize=(10, 6))
ratings['rating'].hist(bins=10)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
st.pyplot(plt)

