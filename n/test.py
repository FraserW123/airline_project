import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
import json

from sklearn.cluster import DBSCAN, KMeans
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

# Take a random sample of 1000 rows
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
print(f"Original number of features standardize: {df_standardize.shape[1]}")
print(f"Original number of features normalized: {df_normalize.shape[1]}")
print(f"Reduced number of features after PCA: {pca.n_components_}")

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

# KMEANS
sil_scores = []
inertia = []
k_range = range(2, 11)

# Fit K-Means for different K values and compute silhouette score
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalize)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df_normalize, kmeans.labels_))

# Plot Silhouette Score vs. K values
plt.figure(figsize=(8, 4))
plt.plot(k_range, sil_scores, marker='o', color='g', label='Silhouette Score')
plt.title('Silhouette Score for Different K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.savefig('silh.png')

# Plot inertia
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o', label='Inertia')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.legend()
plt.savefig("inertia.png")

# DBSCAN

# # Parameter Estimation
# nearest_neighbors = NearestNeighbors(n_neighbors=50)
# neighbors_fit = nearest_neighbors.fit(df_normalize)
# distances, indices = neighbors_fit.kneighbors(df_normalize)

# # Sort distances for the k-th nearest neighbor
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.figure(figsize=(8, 4))
# plt.plot(distances)
# plt.title('k-NN Distance Plot')
# plt.xlabel('Points sorted by distance to k-th nearest neighbor')
# plt.ylabel(f'50-NN Distance')
# plt.savefig("knn.png")

# # Test various min_samples values
# min_samples_values = range(3, 50, 5)  # Adjust the step size based on your needs
# eps_values = np.linspace(0.1, 1.0, 10)

# # Variables to store the best results
# best_eps = None
# best_min_samples = None
# best_score = -1  # Start with the lowest possible silhouette score
# best_labels = None  # To save the labels of the best clustering

# for min_samples in min_samples_values:
#     for eps in eps_values:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         labels = dbscan.fit_predict(df_normalize)
        
#         # Ignore cases where all points are noise
#         if len(set(labels)) <= 1:
#             continue
        
#         sil_score = silhouette_score(df_normalize, labels)

#         # Check if this combination is better than the current best
#         if sil_score > best_score:
#             best_eps = eps
#             best_min_samples = min_samples
#             best_score = sil_score
#             best_labels = labels  # Save the best labels

#         # print(f"min_samples={min_samples}, eps={eps}, silhouette_score={sil_score}")

# # Print the best combination
# print(f"Best eps: {best_eps}")
# print(f"Best min_samples: {best_min_samples}")
# print(f"Best silhouette score: {best_score}")

# dbscan = DBSCAN(eps=1.0, min_samples=13)
# dbscan_labels = dbscan.fit_predict(df_normalize)

# # Check unique clusters
# print("Number of clusters (excluding noise):", len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
# print("Noise points:", sum(dbscan_labels == -1))

# unique_labels = set(dbscan_labels) - {-1}  # Exclude noise points (-1)
# if len(unique_labels) > 1:
#     dbscan_sil_score = silhouette_score(df_normalize, dbscan_labels)
#     print(f"DBSCAN Silhouette Score: {dbscan_sil_score}")
# else:
#     print("Silhouette Score cannot be calculated: Only one cluster found.")

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

data = df_normalize

# Dimensionality Reduction (use PCA or t-SNE)
reduced_data_pca = reduce_dimensions(data, method="pca")  # 2D PCA
reduced_data_tsne = reduce_dimensions(data, method="tsne")  # 2D t-SNE

# Clustering: KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(data)

# Visualize KMeans Results
visualize_cluster(reduced_data_pca[:, 0], reduced_data_pca[:, 1], kmeans_labels, "kmeans_pca.png")
visualize_cluster(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], kmeans_labels, "kmeans_tsne.png")

# Clustering: DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=13)
dbscan_labels = dbscan.fit_predict(data)

# Visualize DBSCAN Results
visualize_cluster(reduced_data_pca[:, 0], reduced_data_pca[:, 1], dbscan_labels, "dbscan_pca.png")
visualize_cluster(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], dbscan_labels, "dbscan_tsne.png")


# Reserve train.csv and test.csv for Benchmarking.
# After building your model, compare its performance against the preprocessed and pre-split data provided in train.csv and test.csv as a sanity check.