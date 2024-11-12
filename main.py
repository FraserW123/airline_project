#%%
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

ontime = df[df['DEP_DEL15'] == 0]
delayed = df[df['DEP_DEL15'] == 1]

# Plot the columns of the ontime and delayed flights and compare, put all the plots in the same figure
# %%

# Create a figure with 5 rows and 6 columns which are well spaced
fig, axs = plt.subplots(5, 6, figsize=(20, 18))


for i, col in enumerate(df.columns):
    ax = axs[i // 6, i % 6]
    ax.hist(ontime[col], alpha=0.5, label='ontime')
    ax.hist(delayed[col], alpha=0.5, label='delayed')
    ax.tick_params(axis='x', rotation=60, labelsize=6)
    ax.set_title(col)
    ax.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('flights.png')
plt.show()


    


# %%
print(ontime.shape)
print(delayed.shape)

print(ontime.columns)

"""
Index(['MONTH', 'DAY_OF_WEEK', 'DEP_DEL15', 'DEP_TIME_BLK', 'DISTANCE_GROUP',
       'SEGMENT_NUMBER', 'CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS',
       'CARRIER_NAME', 'AIRPORT_FLIGHTS_MONTH', 'AIRLINE_FLIGHTS_MONTH',
       'AIRLINE_AIRPORT_FLIGHTS_MONTH', 'AVG_MONTHLY_PASS_AIRPORT',
       'AVG_MONTHLY_PASS_AIRLINE', 'FLT_ATTENDANTS_PER_PASS',
       'GROUND_SERV_PER_PASS', 'PLANE_AGE', 'DEPARTING_AIRPORT', 'LATITUDE',
       'LONGITUDE', 'PREVIOUS_AIRPORT', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND',
       'CARRIER_HISTORICAL', 'DEP_AIRPORT_HIST', 'DAY_HISTORICAL',
       'DEP_BLOCK_HIST'],
      dtype='object')
"""




# %%
