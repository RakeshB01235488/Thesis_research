import util
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cs1.csv")
columns_data = util.cols(data)
column_name = 'runoff_obs'
df_series1 = data[column_name]

anamolies = util.extreme_event_detection(df_series1, column_name)
util.visualize(df_series1, anamolies)