import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Function to generate synthetic data for demonstration purposes
def generate_data(n_samples=100, n_features=4):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    return X

# Sidebar options for user input
st.sidebar.header("User Input Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
n_pca_components = st.sidebar.slider("PCA Components", 2, 4, 2)

# Main panel
st.title("Interactive Clustering Dashboard")

# Generate synthetic data or you can load your dataset here
data = generate_data()

# Data Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# PCA
pca = PCA(n_components=n_pca_components)
pca_data = pca.fit_transform(scaled_data)

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(pca_data)
labels = kmeans.labels_

# PCA Visualization
st.subheader("PCA Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
st.pyplot(fig)

# Cluster Performance Metrics (e.g., inertia)
st.subheader("Cluster Performance Metrics")
st.write(f"Inertia: {kmeans.inertia_}")

# Feature Importance Visualization
st.subheader("Feature Importance")
feature_importance = np.std(scaled_data, axis=0)
sns.barplot(x=feature_importance, y=[f'Feature {i}' for i in range(1, scaled_data.shape[1] + 1)])
plt.xlabel('Standard Deviation')
plt.ylabel('Feature')
st.pyplot(plt.gcf())

# Clustering Interface
st.subheader("Clustering Interface")
new_data = np.array([st.sidebar.number_input(f"Feature {i+1}", 0.0, 1.0, 0.5) for i in range(scaled_data.shape[1])]).reshape(1, -1)
new_scaled_data = scaler.transform(new_data)
new_pca_data = pca.transform(new_scaled_data)
predicted_cluster = kmeans.predict(new_pca_data)
st.write(f"Predicted Cluster: {predicted_cluster[0]}")

# Comparative Analysis
st.subheader("Comparative Analysis")
comparison_df = pd.DataFrame({
    'PCA1': pca_data[:, 0],
    'PCA2': pca_data[:, 1],
    'Cluster': labels
})
st.write(comparison_df)

# Deployment Instructions
st.subheader("Deployment Instructions")
st.write("1. Put your app in a public GitHub repo (and make sure it has a requirements.txt!).")
st.write("2. Sign into share.streamlit.io")
st.write("3. Click 'Deploy an app' and then paste in your GitHub URL.")

# Footer
st.text("Streamlit App by Your Name")
