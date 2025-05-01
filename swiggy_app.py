import pandas as pd 
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib


# Caching data loading functions to optimize performance
@st.cache_data
def load_cleaned_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_encoded_data(file_path):
    return joblib.load(file_path, mmap_mode='r')

# Caching K-Means clustering to avoid recomputation
@st.cache_data
def kmeans_clustering(encoded_data, n_clusters=5):
    # Select only numeric columns for clustering
    numeric_data = encoded_data.select_dtypes(include=[np.number])

    # Scale the data before clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    encoded_data['Cluster'] = kmeans.fit_predict(scaled_data)

    return encoded_data, kmeans

# Caching merge function to avoid recomputing merge each time
@st.cache_data
def merge_data(cleaned_data, encoded_data_with_clusters):
    return pd.merge(cleaned_data, encoded_data_with_clusters[['Cluster', 'name']], on='name', how='left')

# Load cleaned data and encoded data
cleaned_data = load_cleaned_data(r'D:/python_vs/swiggy_recommendation/cleaned_data.csv')
encoded_data = load_encoded_data(r'D:/python_vs/swiggy_recommendation/encoded_data.joblib')

# Perform K-Means clustering
encoded_data_with_clusters, kmeans_model = kmeans_clustering(encoded_data, n_clusters=5)

# Merge cleaned data with encoded data (clusters)
merged_data = merge_data(cleaned_data, encoded_data_with_clusters)

# Streamlit App
st.header("Swiggy Restaurant Recommendations")
st.subheader("Filter Restaurants Based on Your Preferences")

city = st.selectbox('Select City', merged_data['city'].unique())

cuisine = st.selectbox('Select Cuisine', merged_data['cuisine'].unique())

rating = st.slider('Select Rating Range', 1.0, 5.0, (1.0, 5.0), step=0.5)

cost = st.slider('Select Cost Range', 0, 8000, (0, 8000), step=200)

rating_count = st.slider('Select Rating Count Range', 0, 10000, (0, 10000), step=500)

# Apply filters
filtered_data = merged_data[ 
            (merged_data['city'] == city) & 
            (merged_data['cuisine'].str.contains(cuisine, case=False)) & 
            (merged_data['rating'] >= rating[0]) & (merged_data['rating'] <= rating[1]) & 
            (merged_data['cost'] >= cost[0]) & (merged_data['cost'] <= cost[1]) & 
            (merged_data['rating_count'] >= rating_count[0]) & (merged_data['rating_count'] <= rating_count[1]) 
        ]

st.write(filtered_data)