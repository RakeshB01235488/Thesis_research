import util
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cs1.csv")
columns_data = util.cols(data)
column_name = 'runoff_obs'
df_series1 = data[column_name]

anomalies_output = util.extreme_event_detection(df_series1)

binary_output_list = util.anomaly_binary_list(anomalies_output[1])
util.visualize(df_series1, anomalies_output[0])

print(binary_output_list) #list of binary values, 1 if its anomaly, 0 for a noraml value
print(anomalies_output[0]) #outputs the (id, value, 1) for every anomaly observation
