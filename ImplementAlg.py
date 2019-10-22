import math
import numpy as np
import pandas as pd

data = pd.read_csv("cs1.csv")
df = data['runoff_obs']
x = df.values
q = float(input("Enter your desired percentile:"))

#x is an array of observations
#q is desired percentile
#t is initial threshold 
#n_t are values over initial threshold 
#y_t are excess over threshold
#no of peaks over threshold 
#no of observations #updated threshold

	
def pot_kdd_paper(x, q):
   
  t = np.percentile(x,q*100) 
  n_t = [n for n in x if n > t] 
  y_t = [n-t for n in n_t] 
  y_mean = sum(y_t)/float(len(y_t))
  y_min = min(y_t)
  x_star = 2*(y_mean - y_min)/(y_min)**2

  total = 0
  for i in y_t:
    total = total + math.log(1+x_star*i) 
	
  v_x = 1 + (1/len(n_t))*total
  gam = v_x - 1
  sig = gam/float(x_star)
  no_t = len(n_t) 
  n = len(x) 
  z_q = t + (sig/gam)*(((q*n/no_t)**(-gam))-1) 
  print("Initial threshold:", t)
  print("Updated threshold:", z_q)


pot_kdd_paper(x, q)






