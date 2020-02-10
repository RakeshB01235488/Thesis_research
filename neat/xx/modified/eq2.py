import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cs1.csv")
columns = list(data.columns)

df = data['runoff_obs']
#x_values = df.values 
print(cols)
column_name = input("select your desired column")
df = data[column_name]

def extreme_event_detection(q, d):
  A = [] #set of anomalies
  wStarX = [] #window
  m_avg = 0 #moving average of the window's next observation
  k = 0  
  df_200 = df[:200] #first 200 values of the data
  #df1 = data_200['runoff_obs']
  #x_200 = df1.values

  out = pot_kdd_paper(df_200.values, q) #applying pot_kdd_paper method defined in summation.py 
  z_q = out[0]
  t = out[1]
  N_t = out[2]
  y_t = out[3]
  n = out[4]

#removing the abnormal values from the first 100 values,
#i.e., obtaining normal values array
  df_200_norm = df_200[(df_200 <= z_q)]
  #df2 = data_200_norm['runoff_obs']
  #x_200_norm =  df2.values


#obtain first d normal values from the normal values array of first 100 observations
  x_win = df_200_norm.values[:d]
#assign x_win to our original window
  wStarX.extend(x_win)
#compute the moving average of the next observation after the window 
  m_avg = movAvg(wStarX)
#variable change array i.e, x'_i = x_i - m_i
  xd = []

  for i in range (d, d+n):	
	  xd.append(df.values[i] - m_avg)
	  wStarX = np.delete(wStarX, 0)
	  wStarX = np.append(wStarX, df.values[i])
	  m_avg = movAvg(wStarX)

  out1 = pot_kdd_paper(xd, q)
  z_q = out1[0]
  t = out1[1]
  print("threshold (t):", t)
  N_t = out1[2]
  y_t = out1[3]
  n = out1[4] 

  k = n

  for i in range(d+n, len(df.values)):   #xd[n] to xd[len(x_l99)-d-1]
      xd = np.append(xd, df.values[i] - m_avg) #variable change
      if xd[-1] >= z_q: #anomaly case
    	  A.append((i,df.values[i]))#adding anomaly to the anomalies set
    	  m_avg = movAvg(wStarX) 
      elif t <= xd[-1] < z_q: #real peak case
    	  print(xd[-1])
    	  #print("checkpoint_1")
    	  y_i = xd[-1]-t #excess over threshold value
    	  y_t.insert(-1, y_i) #adding to peak set
    	  N_t = N_t+1 #increment the numberof peaks
    	  k = k+1 
    	  pot_kdd_paper(xd, q) #??
    	  z_q = ans([0])
    	  t = ans([1])
    	  wStarX = np.delete(wStarX, 0)
    	  wStarX = np.append(wStarX, df.values[i])
    	  m_avg = movAvg(wStarX) #update of local model
      else: #normal value case
        k = k + 1
        wStarX = np.delete(wStarX, 0)
        wStarX = np.append(wStarX, df.values[i])
        m_avg = movAvg(wStarX) #update of local model


  
  ext = np.array(A)
  X_ext = ext[:,0]
  y_ext = ext[:,1]
  plt.plot(X_ext, y_ext, 'd', color= 'r')

  y_value = df.values
  x_axis = np.arange(len(y_value))
  plt.plot(x_axis, y_value, '+', color= 'b')
  plt.show()



#moving average function
def movAvg(wStarX):
    return np.mean(wStarX)

def pot_kdd_paper(x, q):
  
  t = np.percentile(x,q)
  n_t = [i for i in x if i > t] 
  y_t = [i-t for i in n_t] 
  N_t = len(n_t)
  y_mean = float(sum(y_t))/(len(y_t))
  y_min = min(y_t)
  x_star = 2*(y_mean - y_min)/(y_min)**2
    
  total = 0
  for i in y_t:
    total = total + math.log(1+x_star*i)
    v_x = 1 + (1/len(n_t))*total
    gam = v_x - 1
    sig = gam/float(x_star)
    n = len(x)
    z_q = t + (sig/gam)*(((q*n/N_t)**(-gam))-1)
    #print("Updated threshold:", z_q)
    return (z_q, t, N_t, y_t, n)


extreme_event_detection(95, 30)