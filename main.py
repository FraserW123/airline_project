import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import requests
import json

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import sklearn




# Print the data
df = pd.read_csv("test.csv.xz", compression='xz')
print(df.head())

# take a random sample of 1000 rows
df = df.sample(n=1000)

# Normalize the data

# Select only numeric columns for scaling
# numeric_cols = df.select_dtypes(include=[np.number]).columns
# scaler = StandardScaler()
# scaler.fit(df[numeric_cols])
# df_normalized = scaler.transform(df[numeric_cols])


# # Perform endcoding for the categorical data

# encoder = OneHotEncoder()
# encoder.fit(df)
# df_encoded = encoder.transform(df)

# Create a figure with a 5x6 grid of subplots
fig, axs = plt.subplots(5, 6, figsize=(10, 8))

ontime = df[df['DEP_DEL15'] == 0]
delayed = df[df['DEP_DEL15'] == 1]

# Plot the columns of the ontime and delayed flights and compare, put all the plots in the same figure

for i, col in enumerate(df.columns):
    ax = axs[i // 6, i % 6]
    ax.hist(ontime[col], alpha=0.5, label='ontime')
    ax.hist(delayed[col], alpha=0.5, label='delayed')
    ax.set_title(col)
    ax.legend()
plt.savefig('flights.png')
plt.show()


    

