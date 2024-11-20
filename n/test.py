import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
import json

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import sklearn

# read dataset and check
df = pd.read_csv("full_data_flightdelay.csv.xz", compression='xz')

# Take a random sample of 1000 rows
df = df.sample(n=1000, random_state=42) # use random_state for reproducibility

# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())

# # print(df_sample['CARRIER_NAME'].nunique())
# # print(df_sample['DEPARTING_AIRPORT'].nunique())
# # print(df_sample['PREVIOUS_AIRPORT'].nunique())
# # print(df_sample['DEP_TIME_BLK'].nunique())

# Check for missing values
# print(df_sample.isnull().sum())

# Perform endcoding for the categorical data
# ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DEP_TIME_BLK']
# NOTE: Target encoding

# df = pd.get_dummies(df, columns=['DEP_TIME_BLK', 'CARRIER_NAME', 
#                                                 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT'], drop_first=False)

le = LabelEncoder()
df = df.copy()

def clean_labels_encoder(list_of_labels, df):
    for label in list_of_labels:
        df[label] = le.fit_transform(df[label])
    return df

# clean the labels
list_of_labels = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DEP_TIME_BLK']
df = clean_labels_encoder(list_of_labels, df)

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# fill the missing values with mean
df.fillna(df.mean(), inplace=True)

df_clustering = df.drop(columns=['DEP_DEL15']) 

# Normalize the data
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_clustering)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
principal_components = pca.fit_transform(df_normalized)

# Convert the normalized data back to a DataFrame
df_normalized = pd.DataFrame(df_normalized, columns=df_clustering.columns)

# Check results
print(f"Original number of features: {df_normalized.shape[1]}")
print(f"Reduced number of features after PCA: {pca.n_components_}")

# show correlation
# print(df.corr())

# show the correlation in a plt figure
def show_correlation(df):
    plt.figure(figsize=(20, 10))
    sns.set_theme(style='whitegrid', context='notebook')
    cols = [0, 1, 2]
    sns.heatmap(df.corr(), annot=True, square=False, cmap='coolwarm')
    plt.savefig('corr.png')

# show the correlation
show_correlation(df)

#######################################
# 4. Clustering

# KMEANS
sil_scores = []
k_range = range(2, 11)

# Fit K-Means for different K values and compute silhouette score
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(df_normalized)
    sil_score = silhouette_score(df_normalized, clusters)
    sil_scores.append(sil_score)

# Evaluate with Silhouette Score
# Plot Silhouette Score vs. K values
plt.figure(figsize=(8, 6))
plt.plot(k_range, sil_scores, marker='o', color='g')
plt.title('Silhouette Score for Different K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.savefig('silh.png')

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(df_normalized)
dbscan_labels = dbscan.labels_

# Reserve train.csv and test.csv for Benchmarking.
# After building your model, compare its performance against the preprocessed and pre-split data provided in train.csv and test.csv as a sanity check.