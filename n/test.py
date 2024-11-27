import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

# read dataset and check
df = pd.read_csv("full_data_flightdelay.csv.xz", compression='xz')

# Take a random sample of n rows
df = df.sample(n=10000, random_state=42) # use random_state for reproducibility

le = LabelEncoder()

def clean_labels_encoder(list_of_labels, df):
    for label in list_of_labels:
        df[label] = le.fit_transform(df[label])
    return df

# clean the labels
list_of_labels = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DEP_TIME_BLK']
df = clean_labels_encoder(list_of_labels, df)

# fill the missing values with mean
df.fillna(df.mean(), inplace=True)

# Drop target variable
df_clustering = df.drop(columns=['DEP_DEL15']) 

# Normalize/Standarize the data
standard_scaler = StandardScaler()
minmax_scalar = MinMaxScaler()
df_standardize = standard_scaler.fit_transform(df_clustering)
df_normalize = minmax_scalar.fit_transform(df_clustering)

# Convert the scaled data back to a DataFrame
df_standardize = pd.DataFrame(df_standardize, columns=df_clustering.columns)
df_normalize = pd.DataFrame(df_normalize, columns=df_clustering.columns )

#######################################
# 4. Clustering
def visualize_cluster(x, y, clustering, filename="scatter_plot.png"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=clustering, cmap='coolwarm', s=5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Visualization')
    plt.savefig(filename)
    plt.close()
    return 0

def reduce_dimensions(data, method="pca", n_components=2):
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Unsupported dimensionality reduction method!")
    return reducer.fit_transform(data)

sil_scores = []
inertia = []
k_range = range(2, 11)

# Fit K-Means for different K values
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalize)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df_normalize, kmeans.labels_))

# silhouette_results = pd.DataFrame(sil_scores, columns=['silhouette_scores'])
# inertia_results = pd.DataFrame(inertia, columns=['inertia'])

# Plot inertia (Elbow method suggests k=6)
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o', label='Inertia')
plt.xlabel("Number of Clusters")
plt.ylabel("Distance (squared) to Cluster Center")
plt.title("K-Means Elbow Method")
# plt.savefig('inertia.png')

# Plot Silhouette Score (Silhouette Score suggests k=3)
plt.figure(figsize=(8, 4))
plt.plot(k_range, sil_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Score for Different K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
# plt.savefig('silh.png')

# Dimensionality Reduction (use PCA or t-SNE)
reduced_data_pca = reduce_dimensions(df_normalize, method="pca")  # 2D PCA
reduced_data_tsne = reduce_dimensions(df_normalize, method="tsne")  # 2D t-SNE

# Clustering: KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(df_normalize)

# df_normalize['cluster'] = kmeans.fit_predict(df_normalize)
# print(df_normalize.cluster.value_counts())

# Visualize KMeans Results
visualize_cluster(reduced_data_pca[:, 0], reduced_data_pca[:, 1], kmeans_labels, "2-kmeans_pca.png")
visualize_cluster(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], kmeans_labels, "2-kmeans_tsne.png")

# Hierarchical Clustering (Agglomerative Clustering)

# Compute the linkage matrix
Z = linkage(df_normalize, method='ward')  # 'ward' minimizes the variance of clusters

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
# plt.savefig('dengdrogram.png')

# Based on dendrogram use n_clusters = 2 or 3
agg_clustering = AgglomerativeClustering(n_clusters=2)
agg_labels = agg_clustering.fit_predict(df_normalize)

# Dimensionality Reduction (use PCA or t-SNE)
reduced_data_pca_HC = reduce_dimensions(df_normalize, method="pca")
reduced_data_tsne_HC = reduce_dimensions(df_normalize, method="tsne")

# Visualize Hierarchical Clustering Results
visualize_cluster(reduced_data_pca_HC[:, 0], reduced_data_pca_HC[:, 1], agg_labels, filename="2-agglomerative_pca.png")
visualize_cluster(reduced_data_tsne_HC[:, 0], reduced_data_tsne_HC[:, 1], agg_labels, filename="2-agglomerative_tsne.png")

# Evaluate Clustering Performance

# Store the clustering performance metrics
kmeans_metrics = []
agg_metrics = []

# K-Means Clustering Evaluation
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalize)
    kmeans_labels = kmeans.labels_
    
    # Evaluate performance metrics for K-Means
    kmeans_metrics.append({
        "K": k,
        "Algorithm": "K-Means",
        "Silhouette Score": silhouette_score(df_normalize, kmeans_labels),
        "Calinski-Harabasz Index": calinski_harabasz_score(df_normalize, kmeans_labels),
        "Davies-Bouldin Index": davies_bouldin_score(df_normalize, kmeans_labels)
    })

# Evaluate performance metrics for Agglomerative Clustering
agg_metrics.append({
    "K": 2, 
    "Algorithm": "Agglomerative",
    "Silhouette Score": silhouette_score(df_normalize, agg_labels),
    "Calinski-Harabasz Index": calinski_harabasz_score(df_normalize, agg_labels),
    "Davies-Bouldin Index": davies_bouldin_score(df_normalize, agg_labels)
})

# Convert the metrics lists to DataFrames
kmeans_df = pd.DataFrame(kmeans_metrics)
agg_df = pd.DataFrame(agg_metrics)

# Combine both DataFrames
clustering_metrics_df = pd.concat([kmeans_df, agg_df], ignore_index=True)

#################################################################
# 5. Outlier Detection

from scipy.stats import zscore
# To obtain an approximation of the fraction of anomalous data points in the dataset
# the estimate for contamination is calculated by dividing the total number of outliers by the total number of samples

z_scores = np.abs(zscore(df_normalize))
outliers_count = (z_scores > 3).max(axis=1).sum()
contamination_estimate = outliers_count / df_normalize.shape[0]

# Print the result
print(f"Estimated contamination: {contamination_estimate:.2f}")

# Isolation Forest
iso_forest = IsolationForest(contamination=0.13, random_state=42)
outliers_iso = iso_forest.fit_predict(df_normalize)

# Mark outliers (1 = inlier, -1 = outlier)
outliers_iso = (outliers_iso == -1)

visualize_cluster(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], outliers_iso, "isoforest-tsne-0.13.png")
visualize_cluster(reduced_data_pca[:, 0], reduced_data_pca[:, 1], outliers_iso, "isoforest-pca-0.13.png")

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.13)
outliers_lof = lof.fit_predict(df_normalize)

# Mark outliers (1 = inlier, -1 = outlier)
outliers_lof = (outliers_lof == -1)

visualize_cluster(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], outliers_lof, "LOF-tsne-0.13.png")
visualize_cluster(reduced_data_pca[:, 0], reduced_data_pca[:, 1], outliers_lof, "LOF-pca-0.13.png")
