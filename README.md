# Airline Flight Delay Project
## Dataset Selection:

https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations/data

The dataset has almost 6.5 million rows. For better running times, I sampled 10,000 rows from the dataset which will be worked on

## Data Exploration

The dataset contains 26 columns

Index: 10000 entries, 1866883 to 4201534
Data columns (total 26 columns):
| # | Column                        | Non-Null Count  |Dtype   |  
|---| ------                        | --------------  |-----   |  
| 0 |  MONTH                        |  10000 non-null | int64  | 
| 1 |  DAY_OF_WEEK                  |  10000 non-null | int64  | 
| 2 |  DEP_DEL15                    |  10000 non-null | int64  | 
| 3 |  DEP_TIME_BLK                 |  10000 non-null | object | 
| 4 |  DISTANCE_GROUP               |  10000 non-null | int64  | 
| 5 |  SEGMENT_NUMBER               |  10000 non-null | int64  | 
| 6 |  CONCURRENT_FLIGHTS           |  10000 non-null | int64  | 
| 7 |  NUMBER_OF_SEATS              |  10000 non-null | int64  | 
| 8 |  CARRIER_NAME                 |  10000 non-null | object | 
| 9 |  AIRPORT_FLIGHTS_MONTH        |  10000 non-null | int64  | 
| 10|  AIRLINE_FLIGHTS_MONTH        |  10000 non-null | int64  | 
| 11|  AIRLINE_AIRPORT_FLIGHTS_MONTH|  10000 non-null | int64  | 
| 12|  AVG_MONTHLY_PASS_AIRPORT     |  10000 non-null | int64  | 
| 13|  AVG_MONTHLY_PASS_AIRLINE     |  10000 non-null | int64  | 
| 14|  FLT_ATTENDANTS_PER_PASS      |  10000 non-null | float64|
| 15|  GROUND_SERV_PER_PASS         |  10000 non-null | float64|
| 16|  PLANE_AGE                    |  10000 non-null | int64  | 
| 17|  DEPARTING_AIRPORT            |  10000 non-null | object |
| 18|  LATITUDE                     |  10000 non-null | float64|
| 19|  LONGITUDE                    |  10000 non-null | float64|
| 20|  PREVIOUS_AIRPORT             |  10000 non-null | object |
| 21|  PRCP                         |  10000 non-null | float64|
| 22|  SNOW                         |  10000 non-null | float64|
| 23|  SNWD                         |  10000 non-null | float64|
| 24|  TMAX                         |  10000 non-null | float64|
| 25|  AWND                         |  10000 non-null | float64|
dtypes: float64(9), int64(13), object(4)
memory usage: 2.1+ MB

## Here are some statistics of these featues
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH</th>
      <th>DAY_OF_WEEK</th>
      <th>DEP_DEL15</th>
      <th>DEP_TIME_BLK</th>
      <th>DISTANCE_GROUP</th>
      <th>SEGMENT_NUMBER</th>
      <th>CONCURRENT_FLIGHTS</th>
      <th>NUMBER_OF_SEATS</th>
      <th>CARRIER_NAME</th>
      <th>AIRPORT_FLIGHTS_MONTH</th>
      <th>...</th>
      <th>PLANE_AGE</th>
      <th>DEPARTING_AIRPORT</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>PREVIOUS_AIRPORT</th>
      <th>PRCP</th>
      <th>SNOW</th>
      <th>SNWD</th>
      <th>TMAX</th>
      <th>AWND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>...</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.634800</td>
      <td>3.927900</td>
      <td>0.193400</td>
      <td>8.231900</td>
      <td>3.843400</td>
      <td>3.049100</td>
      <td>27.765400</td>
      <td>133.974900</td>
      <td>9.111100</td>
      <td>12640.642100</td>
      <td>...</td>
      <td>11.552000</td>
      <td>42.881500</td>
      <td>36.728235</td>
      <td>-94.237052</td>
      <td>154.376000</td>
      <td>0.106784</td>
      <td>0.028970</td>
      <td>0.101060</td>
      <td>71.490100</td>
      <td>8.306849</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.413385</td>
      <td>1.989296</td>
      <td>0.394984</td>
      <td>4.898584</td>
      <td>2.381353</td>
      <td>1.759318</td>
      <td>21.403191</td>
      <td>46.559062</td>
      <td>5.129248</td>
      <td>8810.857618</td>
      <td>...</td>
      <td>6.927185</td>
      <td>27.093569</td>
      <td>5.552612</td>
      <td>17.871948</td>
      <td>72.541372</td>
      <td>0.343852</td>
      <td>0.307371</td>
      <td>0.779256</td>
      <td>18.198885</td>
      <td>3.605541</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>44.000000</td>
      <td>0.000000</td>
      <td>1100.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>19.739000</td>
      <td>-159.346000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-10.000000</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>11.000000</td>
      <td>90.000000</td>
      <td>5.000000</td>
      <td>5337.000000</td>
      <td>...</td>
      <td>5.000000</td>
      <td>17.000000</td>
      <td>33.436000</td>
      <td>-106.377000</td>
      <td>99.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>59.000000</td>
      <td>5.820000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>23.000000</td>
      <td>143.000000</td>
      <td>10.000000</td>
      <td>11588.000000</td>
      <td>...</td>
      <td>12.000000</td>
      <td>42.000000</td>
      <td>37.363000</td>
      <td>-87.906000</td>
      <td>182.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>74.000000</td>
      <td>7.830000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>39.000000</td>
      <td>172.000000</td>
      <td>14.000000</td>
      <td>17522.000000</td>
      <td>...</td>
      <td>17.000000</td>
      <td>66.000000</td>
      <td>40.779000</td>
      <td>-80.936000</td>
      <td>203.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>86.000000</td>
      <td>10.290000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>11.000000</td>
      <td>13.000000</td>
      <td>108.000000</td>
      <td>337.000000</td>
      <td>16.000000</td>
      <td>35256.000000</td>
      <td>...</td>
      <td>32.000000</td>
      <td>94.000000</td>
      <td>61.169000</td>
      <td>-70.304000</td>
      <td>291.000000</td>
      <td>11.630000</td>
      <td>9.900000</td>
      <td>18.900000</td>
      <td>115.000000</td>
      <td>33.780000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>

### Correlation Heatmap between all features
![](results/heatmap.png)
### Correlation between Delayed Flights and Features
![](results/correlation%20bar%20graph.png)


### Data Proportions
![](results/delayedproportion.png)
### Ontime Flights vs Delayed Flights
![](results/flights.png)

### Departing Airports Delays
Here is a map of the departing airports in the dataset with the noted frequency of flights being delayed
![](results/departing_airport_delays.png)
The top 5 airports with the most delays are 
|DEPARTING_AIRPORT                 | Flight Delays    |
|----------------------------------|------------------|
|Atlanta Municipal                 | 109              |
|Stapleton International           | 108              |
|Chicago O'Hare International      | 97               |
|Dallas Fort Worth Regional        | 96               |
|Douglas Municipal                 | 74               |






