import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Title of the app
st.title("Clustering Visualization with Silhouette Scores")

# Upload CSV data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Display the data
    st.write(data)

    # Select number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 10)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    data['Cluster'] = labels

    # Calculate and display Silhouette Score
    silhouette_avg = silhouette_score(data, labels)
    st.write(f"Silhouette Score: {silhouette_avg}")

    # Visualize the clusters
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
    plt.title(f"KMeans Clustering with {n_clusters} Clusters")
    st.pyplot(plt)
