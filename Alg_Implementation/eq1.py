import math
import numpy as np
import pandas as pd
import pdb

#pdb.set_trace()
data = pd.read_csv("cs1.csv")
df = data['runoff_obs']
x_values = df.values #runoff_obs of whole dataset
q = float(input("Enter your desired percentile:"))

A = [] #set of anomalies
d = int(input("Enter your desired window length:"))
wStarX = [] #window
m_avg = 0 #moving average of the window's next observation
k = 0  


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
  	print("Updated threshold:", z_q)
  	return (z_q, t, N_t, y_t, n)


#first 100 values of the data
data_200 = data[:200]
df1 = data_200['runoff_obs']
x_200 = df1.values

#applying pot_kdd_paper method defined in summation.py 
out = pot_kdd_paper(x_200, q)
z_q = out[0]
t = out[1]
N_t = out[2]
y_t = out[3]
n = out[4]

#removing the abnormal values from the first 100 values,
#i.e., obtaining normal values array
data_200_norm = data_200[(data_200.runoff_obs <= z_q)]
df2 = data_200_norm['runoff_obs']
x_200_norm =  df2.values


#obtain first d normal values from the normal values array of first 100 observations
x_win = x_200_norm[:d]
###x_win_d = data[(data.runoff_obs <= z_q)]
###data_100 = data_L98[:100]
###df1 = data_100['runoff_obs']
###x_100 = df1.values

#assign x_win to our original window
wStarX.extend(x_win)

#compute the moving average of the next observation after the window 
m_avg = movAvg(wStarX)

##print(wStarX)
##print(x_100_norm[:d])
#ans = pot_kdd_paper(x_vals, q)
#z_q = ans[0]
#t = ans[1]
#x_vals_norm = x_vals[x_vals <= z_q]
#wStarX.append(x_vals_norm[:d])
#print(len(wStarX))
#m_avg = movAvg(wStarX)
#print(len(wStarX))

#variable change array i.e, x'_i = x_i - m_i
xd = []

for i in range (d, d+n):	
	xd.append(x_values[i] - m_avg)
	#wStarX = np.delete(wStarX, i-d)
	#wStarX = np.insert(wStarX, d, x_values[i])
	wStarX = np.delete(wStarX, 0)
	wStarX = np.append(wStarX, x_values[i])
	m_avg = movAvg(wStarX)

#print(xd)
#wStarX = np.append(wStarX, x_values[i])	
##print(x_values[i] - m_avg)
#print(len(wStarX))    ###
#print(xd)  
#pot_kdd_paper(xd, q) 
out1 = pot_kdd_paper(xd, q)
z_q = out1[0]
t = out1[1]
print("threshold (t):", t)
N_t = out1[2]
y_t = out1[3]
n = out1[4] 
#print(ans) 
#print(z_q, t)

k = n ### k???

for i in range(d+n, len(x_values)):   #xd[n] to xd[len(x_l99)-d-1]
    xd = np.append(xd, x_values[i] - m_avg) #variable change
    #print(len(xd))
    #print(t, xd[-1])
    if xd[-1] >= z_q: #anomaly case
    	A.append((i,x_values[i]))#adding anomaly to the anomalies set
    	m_avg = movAvg(wStarX)
    	#print("checkpoint_0")
    	#print(A) 
    elif t <= xd[-1] < z_q: #real peak case
    	print(xd[-1])
    	print("checkpoint_1")
    	y_i = xd[-1]-t #excess over threshold value
    	y_t.insert(-1, y_i) #adding to peak set
    	N_t = N_t+1 #increment the numberof peaks
    	k = k+1 
    	#xd = xd[:-1]###
    	pot_kdd_paper(xd, q) #??
    	z_q = ans([0])
    	t = ans([1])
    	#wStarX = np.append(wStarX[1:], x_values[i])#window slide
    	wStarX = np.delete(wStarX, 0)
    	wStarX = np.append(wStarX, x_values[i])
    	m_avg = movAvg(wStarX) #update of local model
    else: #normal value case
        k = k + 1
        #print("checkpoint_2")
        #wStarX = np.append(wStarX[1:], x_values[i])
        wStarX = np.delete(wStarX, 0)
        wStarX = np.append(wStarX, x_values[i])
        m_avg = movAvg(wStarX) #update of local model


import matplotlib.pyplot as plt
ext = np.array(A)
# print(ext[:5])
X_ext = ext[:,0]
y_ext = ext[:,1]
plt.plot(X_ext, y_ext, 'd', color= 'r')

y_value = x_values
x_axis = np.arange(len(y_value))
plt.plot(x_axis, y_value, '+', color= 'b')
plt.show()
# plt.plot()

#print(A)
#print(len(A))
        #wStarX = np.append(wStarX[1:], x_values[i])
        #print(len(wStarX))
          #no window update


#def movAvg(i):
#temp = 0
#for k in range (1,d)
#   temp = temp + (1/d)(x_star_(i-k))
#   movAvgi = temp
#   return movAvgi


