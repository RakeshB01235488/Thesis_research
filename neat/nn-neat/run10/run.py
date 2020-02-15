
import numpy as np
import pandas as pd
import os
import neat
import pickle
import timeit
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy import array
from sklearn.metrics import r2_score


# 2-input XOR inputs and expected outputs.

df1 = pd.read_csv('../combine_partial_NoLag-1.csv')
df2 = pd.read_csv('../combine_partial_NoLag-2.csv')

#df1.pop('Date')
#df1.pop('m0')


#df2.pop('Date')
#df2.pop('m0')
#print(df1.head())
#print(df2.head())

#print(df1.shape)
#print(df2.shape)
#print(df1.columns)
#print(df2.columns)


y_train = df1.pop('East_12')
y_test = df2.pop('East_12')

X_train = df1
X_test = df2

print(X_train.shape)
print(X_test.shape)
print(X_train.columns)
print(X_test.columns)


X_train = X_train.values.tolist()
y_train = y_train.values.tolist()
X_test = X_test.values.tolist()
y_test = y_test.values.tolist()


def eval_genomes(genomes, config): #function Used for training model 
# using the training set
    for genome_id, genome in genomes:
        genome.fitness = -1
        net = neat.nn.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(X_train, y_train):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo) ** 2 #Distance from 
            # the correct output summed for all 84 inputs patterns

for i in range(1,1+1):
    s1 = timeit.default_timer()  

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(100))
    
    # Run until a solution is found.
    winner = p.run(eval_genomes, 500)  # run for 12000 to test 
    with open('winner_genome'+str(i), 'wb') as f:
        pickle.dump(winner, f)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Make and show prediction on unseen data (test set) using winner NN's 
    # genome.
    print('\nOutput:')
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    list2 = []
    # print ('type(list2)',type(list2))

    y_test = array(y_test)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    for xi, xo in zip(X_test, y_test):
        output = winner_net.activate(xi)
        # print ('type(output)',type(output))
        list2.append(output)
        # print("  input {!r}, expected output {!r}, got {!r}".format(
        # xi, xo, output))
    pred = np.array(list2)
    print('pred',pred)
    np.savetxt("pred-ge"+str(i)+".txt",pred)
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
    #y_test = y_test.ix[range(0,N1),['l0']]
    yhat =pred
    #print("yhat")
    y= y_test
    # rmse = sqrt(mean_squared_error(y, yhat))
    # print("GE RMSE=",rmse)
    # print("GE NRMSE=",100*rmse/max(y))
    # mae= mean_absolute_error(y, yhat)
    # print("GE MAE=",mae)
    # print("GE NMAE=",100*mae/max(y))
    rsq = r2_score(y, yhat)
    print("RSquared=",rsq)
print ('Runing time is Hour:',round((s2 -s1)/3600,2))
"""
"""


