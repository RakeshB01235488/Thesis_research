import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cs1.csv")


def extreme_event_detection(d):
  
  columns = list(data.columns)
  print(columns)
  column_name = input("select your desired column:")
  df = data[column_name]
  A = [] #set of anomalies
  wStarX = [] #window
  m_avg = 0 #moving average of the window's next observation
  k = 0  
  df_200 = df[:200] #first 200 values of the data
  
  out = pot_kdd_paper(df_200.values) #applying pot_kdd_paper method defined in summation.py 
  z_q = out[0]
  #print(z_q)
  t = out[1]
  #print(t)
  N_t = out[2]
  y_t = out[3]
  n = out[4]

  df_200_norm = df_200[(df_200 <= z_q)]
  #obtain first d normal values from the normal values array of first 100 observations
  x_win = df_200_norm.values[:d]
  #assign x_win to our original window
  wStarX.extend(x_win)
  #compute the moving average of the next observation after the window 
  m_avg = movAvg(wStarX)
  #variable change array i.e, x'_i = x_i - m_i
  xd = []
  #m_avg = []

  for i in range (d, d+200):	
	  xd.append(abs(df.values[i] - m_avg))
	  wStarX = np.delete(wStarX, 0)
	  wStarX = np.append(wStarX, df.values[i])
	  m_avg = movAvg(wStarX)
	  #print(m_avg)

  #print(xd)
  #print(z_q)
  out1 = pot_kdd_paper(xd)
  z_q = out1[0]
  #print(z_q)
  t = out1[1]
  #print(t)
  N_t = out1[2]
  y_t = out1[3]
  n = out1[4] 

  #print("updated threshold:", z_q)
  
  #print(z_q)
  k = n

  for i in range(d+200, len(df.values)):   #xd[n] to xd[len(x_l99)-d-1]
      xd = np.append(xd, df.values[i] - m_avg)
      #variable change
      #print(xd[-1])
      if xd[-1] >= z_q: #anomaly case
    	  A.append((i,df.values[i]))#adding anomaly to the anomalies set
    	  m_avg = movAvg(wStarX)
    	  #print(A) 
      
      #elif t <= xd[-1] < z_q: #real peak case
      elif xd[-1] > t:	
    	  #print(xd[-1])
    	  #print("checkpoint_1")
    	  # y_i = xd[-1]-t #excess over threshold value
    	  # y_t.insert(-1, y_i) #adding to peak set
    	  # N_t = N_t+1 #increment the number of peaks
    	  k = k+1 
    	  ans = pot_kdd_paper(xd) #??
    	  z_q = ans[0]
    	  t = ans[1]
    	  N_t = ans[2]
    	  y_t = ans[3]
    	  n = ans[4] 

    	  wStarX = np.delete(wStarX, 0)
    	  wStarX = np.append(wStarX, df.values[i])
    	  m_avg = movAvg(wStarX) #update of local model
      else: #normal value case
        k = k + 1
        wStarX = np.delete(wStarX, 0)
        wStarX = np.append(wStarX, df.values[i])
        m_avg = movAvg(wStarX) #update of local model


  print(A)
  print(len(A))
  # ext = np.array(A)
  # X_ext = ext[:,0]
  # y_ext = ext[:,1]
  # plt.plot(X_ext, y_ext, 'd', color= 'r')

  # y_value = df.values
  # x_axis = np.arange(len(y_value))
  # plt.plot(x_axis, y_value, '+', color= 'b')
  # plt.show()



#moving average function
def movAvg(wStarX):
    return np.mean(wStarX)

def pot_kdd_paper(x):
  
  t = np.percentile(x,95)
  n_t = [i for i in x if i > t] 
  y_t = [i-t for i in n_t] 
  #print(y_t)
  N_t = len(n_t)
  #print(N_t)
  y_mean = float(sum(y_t))/(len(y_t))
  #print(y_mean)
  y_min = min(y_t)
  #print(y_min)
  x_star = 2*(y_mean - y_min)/(y_min)**2
  #print(x_star)  
  total = 0
  for i in y_t:
    total = total + math.log(1+x_star*i)

  v_x = 1 + (1/len(n_t))*total
  #print(v_x)
  gam = v_x - 1
  #print(gam)
  sig = gam/float(x_star)
  #print(sig)
  n = len(x)
  asd = ((0.02*n)/N_t)**(-gam)
  #print(asd)
  z_q = t + sig*(asd-1)/gam
  #print("Updated threshold:", z_q)
  return (z_q, t, N_t, y_t, n)


extreme_event_detection(300)