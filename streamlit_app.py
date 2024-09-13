import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import altair as alt

# Title of the app
st.title("Consumer Purchase Behavior Based on Avocado Types")

# Load the dataset
data = pd.read_csv("df_encoded.csv")
df = pd.DataFrame(data)

# Define a function for Agglomerative Clustering
def agglomerative_clustering(n_clusters):
    X_cluster = filtered_df[['type_conventional', 'type_organic', 'AveragePrice', 'Total Volume']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Reduce dimensions to 2D using PCA before clustering
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for PCA components
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

    # Create a scatter plot using Altair with labeled axes
    scatter_plot = alt.Chart(pca_df).mark_circle().encode(
        x=alt.X('PC1', title='Principal Component 1'),
        y=alt.Y('PC2', title='Principal Component 2')
    ).properties(
        width=600,
        height=400
    )

    # Display the plot in Streamlit
    st.altair_chart(scatter_plot)

    # Perform Agglomerative clustering
    agglo_model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglo_model.fit_predict(X_pca)

    # Add cluster labels to the DataFrame
    filtered_df['cluster'] = labels
    num_clusters = len(np.unique(filtered_df['cluster']))
    st.write("Number of clusters:", num_clusters)

    # Plot the clustering results in 2D
    plt.figure(figsize=(10, 6))

    unique_clusters = np.unique(filtered_df['cluster'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    for cluster_id, color in zip(unique_clusters, colors):
        plt.scatter(
            X_pca[filtered_df['cluster'] == cluster_id, 0],
            X_pca[filtered_df['cluster'] == cluster_id, 1],
            c=[color],
            label=f'Cluster {cluster_id}',
            alpha=0.7, s=50
        )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Agglomerative Clustering (PCA-reduced 2D)')
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot in Streamlit
    st.pyplot(plt)

    silhouette_avg = silhouette_score(X_pca, labels)
    st.write("Silhouette Score:", round(silhouette_avg, 2))

# Define a function for Gaussian Mixture Model (GMM) Clustering
def gmm_clustering(n_components):
    # Select features for clustering
    X_cluster = filtered_df[['type_conventional', 'type_organic', 'AveragePrice', 'Total Volume']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Reduce dimensions to 2D using PCA before clustering
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for PCA components
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

    # Create a scatter plot using Altair with labeled axes
    scatter_plot = alt.Chart(pca_df).mark_circle().encode(
        x=alt.X('PC1', title='Principal Component 1'),
        y=alt.Y('PC2', title='Principal Component 2')
    ).properties(
        width=600,
        height=400
    )

    # Display the plot in Streamlit
    st.altair_chart(scatter_plot)

    # Perform GMM clustering
    gmm_model = GaussianMixture(n_components=n_components)
    labels = gmm_model.fit_predict(X_pca)

    # Add cluster labels to the DataFrame
    filtered_df['cluster'] = labels
    num_clusters = len(np.unique(filtered_df['cluster']))
    st.write("Number of clusters:", num_clusters)

    # Plot the clustering results in 2D
    plt.figure(figsize=(10, 6))

    unique_clusters = np.unique(filtered_df['cluster'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    for cluster_id, color in zip(unique_clusters, colors):
        plt.scatter(
            X_pca[filtered_df['cluster'] == cluster_id, 0],
            X_pca[filtered_df['cluster'] == cluster_id, 1],
            c=[color],
            label=f'Cluster {cluster_id}',
            alpha=0.7, s=50
        )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('GMM Clustering (PCA-reduced 2D)')
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot in Streamlit
    st.pyplot(plt)

    silhouette_avg = silhouette_score(X_pca, labels)
    st.write("Silhouette Score:", round(silhouette_avg, 2))

# Auto-select the purpose
st.write("Purpose: Consumer Purchase Behavior Based on Avocado Types")

# Cluster selection based on the algorithms you specified
cluster = st.selectbox("Cluster", ("Agglomerative", "GMM"))

# Year range filter using a slider
year_range_filter = st.slider('Select Year Range', min_value=2015, max_value=2018, value=(2015, 2018))

# Filter the DataFrame based on the selected year range
filtered_df = df[(df['year'] >= year_range_filter[0]) & (df['year'] <= year_range_filter[1])]

# Display clustering parameters based on the selected clustering algorithm
if cluster == "Agglomerative":
    n_clusters = st.slider('Number of Clusters', 2, 10, 5, step=1)
    agglomerative_clustering(n_clusters)
    
elif cluster == "GMM":
    n_components = st.slider('Number of Components', 2, 10, 5, step=1)
    gmm_clustering(n_components)
