import math
import numpy as np
import pandas as pd
import pdb
from summation import pot_kdd_paper

#pdb.set_trace()

data = pd.read_csv("cs1.csv")
df = data['runoff_obs']
x_values = df.values
q = float(input("Enter your desired percentile:"))

A = []
d = 100
wStarX = []
m_avg = 0
t = np.percentile(x_values,q*100)
n_t = [n for n in x_values if n > t] 
y_t = [n-t for n in n_t] 
N_t = len(n_t) 

k = 0

def movAvg(wStarX):
    return np.mean(wStarX)

z_q = pot_kdd_paper(x_values, q, k, N_t, t) 
data_L99 = data[(data.runoff_obs <= z_q)]
df1 = data_L99['runoff_obs']
x_L99 = df1.values
wStarX.append(x_L99[:d])
m_avg = movAvg(wStarX)
xd = []
n = 500

for i in range (d+1, d+n):
    xd.append(x_L99[i] - m_avg)
    wStarX = np.append(wStarX[1:], x_L99[i])
    m_avg = movAvg(wStarX)
    
t = np.percentile(xd,q*100)
n_t = [n for n in xd if n > t] 
y_t = [n-t for n in n_t] 
N_t = len(n_t)
k = 0  
pot_kdd_paper(xd, q, k, N_t, t)  

k = n
for i in range(d+n+1, len(x_L99)):
    xd.append(x_L99[i] - m_avg)
    if xd[-1] > z_q:
        A.append((i,x_L99[i]))
        wStarX = np.append(wStarX[1:], x_L99[i])
        m_avg = movAvg(wStarX)
    elif xd[-1] > t:
        y_t = list.append(xd[i] - t) 
        N_t = N_t + 1
        k = k + 1
        pot_kdd_paper(xd, q, k, N_t, t)
        wStarX = np.append(wStarX[1:], x_L99[i])
        m_avg = movAvg(wStarX)
    else:
        k = k + 1
        wStarX = np.append(wStarX[1:], x_L99[i])
        m_avg = movAvg(wStarX)
        


#def movAvg(i):
#temp = 0
#for k in range (1,d)
#   temp = temp + (1/d)(x_star_(i-k))
#   movAvgi = temp
#   return movAvgi


