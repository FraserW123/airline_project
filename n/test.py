import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
import json

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import sklearn

# read dataset and check
df = pd.read_csv("full_data_flightdelay.csv.xz", compression='xz')

# Take a random sample of n rows
df = df.sample(n=1000, random_state=42) # use random_state for reproducibility

##########################
## INITIAL DATA EXPLORATION

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

##########################

## Checking for unique values in categorical columns

# print(df_sample['CARRIER_NAME'].nunique())
# print(df_sample['DEPARTING_AIRPORT'].nunique())
# print(df_sample['PREVIOUS_AIRPORT'].nunique())
# print(df_sample['DEP_TIME_BLK'].nunique())

##########################

## Check for missing values
# print(df_sample.isnull().sum())

## Perform endcoding for the categorical data
## ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DEP_TIME_BLK']
## NOTE: Target encoding

# df = pd.get_dummies(df, columns=['DEP_TIME_BLK', 'CARRIER_NAME', 
#                                                 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT'], drop_first=False)

le = LabelEncoder()

def clean_labels_encoder(list_of_labels, df):
    for label in list_of_labels:
        df[label] = le.fit_transform(df[label])
    return df

# clean the labels
list_of_labels = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DEP_TIME_BLK']
df = clean_labels_encoder(list_of_labels, df)

## Check updated dataset
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# fill the missing values with mean
df.fillna(df.mean(), inplace=True)

# Drop target variable
df_clustering = df.drop(columns=['DEP_DEL15']) 

# Normalize/Standarize the data
standard_scaler = StandardScaler()
minmax_scalar = MinMaxScaler()
df_standardize = standard_scaler.fit_transform(df_clustering)
df_normalize = minmax_scalar.fit_transform(df_clustering)

# # Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
principal_components_s = pca.fit_transform(df_standardize)
principal_components_n = pca.fit_transform(df_normalize)

# Convert the scaled data back to a DataFrame
df_standardize = pd.DataFrame(df_standardize, columns=df_clustering.columns)
df_normalize = pd.DataFrame(df_normalize, columns=df_clustering.columns )

# # Check results
# print(f"Original number of features standardize: {df_standardize.shape[1]}")
# print(f"Original number of features normalized: {df_normalize.shape[1]}")
# print(f"Reduced number of features after PCA: {pca.n_components_}")

# print(df.head())
# print(df_standardize.head())
# print(df_normalize.head())

# show correlation
# print(df.corr())

# # show the correlation in a plt figure
# def show_correlation(df):
#     plt.figure(figsize=(20, 10))
#     sns.set_theme(style='whitegrid', context='notebook')
#     cols = [0, 1, 2]
#     sns.heatmap(df.corr(), annot=True, square=False, cmap='coolwarm')
#     plt.savefig('corr.png')

# # show the correlation
# show_correlation(df)

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

# KMEANS with Elbow Method
# km = KMeans(n_clusters=2, random_state=42)
# df_normalize['cluster'] = km.fit_predict(df_normalize)
# print(df_normalize.head())
# print(df_normalize.cluster.value_counts())

# Scale data (sd from the mean)
# for col in df.columns:
#     avg = df[col].mean()
#     sd = df[col].std()
#     df[col] = df[col].apply(lambda x: (x-avg)/sd)

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
plt.savefig('inertia.png')

# Plot Silhouette Score (Silhouette Score suggests k=3)
plt.figure(figsize=(8, 4))
plt.plot(k_range, sil_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Score for Different K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.savefig('silh.png')

# Dimensionality Reduction (use PCA or t-SNE)
reduced_data_pca = reduce_dimensions(df_normalize, method="pca")  # 2D PCA
reduced_data_tsne = reduce_dimensions(df_normalize, method="tsne")  # 2D t-SNE

# Clustering: KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(df_normalize)

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
plt.savefig('dengdrogram.png')

# Based on dendrogram use n_clusters = 2 or 3
agg_clustering = AgglomerativeClustering(n_clusters=2)
agg_labels = agg_clustering.fit_predict(df_normalize)

# Dimensionality Reduction (use PCA or t-SNE)
reduced_data_pca_HC = reduce_dimensions(df_normalize, method="pca")
reduced_data_tsne_HC = reduce_dimensions(df_normalize, method="tsne")

# Visualize Hierarchical Clustering Results
visualize_cluster(reduced_data_pca_HC[:, 0], reduced_data_pca_HC[:, 1], agg_labels, filename="agglomerative_pca.png")
visualize_cluster(reduced_data_tsne_HC[:, 0], reduced_data_tsne_HC[:, 1], agg_labels, filename="agglomerative_tsne.png")

agg_sil_score = silhouette_score(df_normalize, agg_labels)
print(f"Silhouette Score for Agglomerative Clustering: {agg_sil_score}")




# Reserve train.csv and test.csv for Benchmarking.
# After building your model, compare its performance against the preprocessed and pre-split data provided in train.csv and test.csv as a sanity check.