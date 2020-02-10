

import numpy as np
import pandas as pd
import os
import neat
import pickle
import timeit
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# 2-input XOR inputs and expected outputs.

df1 = pd.read_csv('../ppall_2009.csv')
df2 = pd.read_csv('../ppall_2010.csv')

df1.pop('l0')
df1.pop('m0')

df2.pop('l0')
df2.pop('m0')
print(df1.head())
print(df2.head())

print(df1.shape)
print(df2.shape)
print(df1.columns)
print(df2.columns)


y_train = df1.pop('sum')
y_test = df2.pop('sum')

X_train = df1
X_test = df2

print(X_train.shape)
print(X_test.shape)
print(X_train.columns)
print(X_test.columns)


X_train = X_train.values.tolist()
y_train = y_train.values.tolist()
X_test = X_test.values.tolist()
y_test = y_test.values


def eval_genomes(genomes, config): #function Used for training model 
# using the training set
    for genome_id, genome in genomes:
        genome.fitness = -1
        net = neat.nn.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(X_train, y_train):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo) ** 2 #Distance from 
            # the correct output summed for all 84 inputs patterns
			
			
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                 'config')
for k in range(1,3+1):
    
    f=open('winner_genome'+str(k),'rb')  
    winner2=pickle.load(f)  

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner2))

    # Make and show prediction on unseen data (test set) using winner NN's 
    # genome.
    print('\nOutput:',k)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    list2 = []
    # print ('type(list2)',type(list2))

    winner_net = neat.nn.RecurrentNetwork.create(winner2, config)
    for xi, xo in zip(X_test, y_test):
        output = winner_net.activate(xi)
        # print ('type(output)',type(output))
        list2.append(output)
        # print("  input {!r}, expected output {!r}, got {!r}".format(
        # xi, xo, output))
    pred = np.array(list2)
    print('pred',pred)

    # np.savetxt("pred-ge"+str(i)+".txt",pred)


    yhat =pred
    y= y_test
    print(type(yhat))
    print(type(y))

    rmse = sqrt(mean_squared_error(y, yhat))

    max_y =300.5
    print('winner:'+str(k))
    print("GE+Mits RMSE=",rmse)
    print("GE+Mits NRMSE=",100*rmse/max_y)
    mae= mean_absolute_error(y, yhat)
    print("GE+Mits MAE=",mae)
    print("GE+Mits NMAE=",100*mae/max_y)


