import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from utilites import gradient_calc, cost_function, training_fn, plot_cost_graph, find_mse
from optimizing_functions import *

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-op","--optimizer",help="Select the optimizer that you want to use for training\n Available are  ['Batch','Stochastic','MiniBatch','Momentum','Adam','Adagrad','Adadelta', 'Nadam','AdaMax','Nesterov','RMSProp']")

args = parser.parse_args()

optimizing_fn_name = args.optimizer + '_optimization'
try:
    optimizing_fn = locals()[optimizing_fn_name]
except KeyError:
    print("\nInvalid optimizer choosen\n")
    print("Select the optimizer that you want to use for training\n Available are  ['Batch','Stochastic','MiniBatch','Momentum','Adam','Adagrad','Adadelta', 'Nadam','AdaMax','Nesterov','RMSprop']")
    quit()
# setting values
# BATCH GRADIENT DESCENT DEFAULTS
if args.optimizer == 'Batch':
    epochs = 12000
    lr = 0.001
    extra_params = {}

# STOCHASTIC GRADIENT DESCENT DEFAULTS
if args.optimizer  == 'Stochastic':
    epochs = 2
    lr = 0.001
    extra_params = {}

# MINI BATCH GRADIENT DESCENT DEFAULTS
if args.optimizer == 'MiniBatch':
    epochs = 5
    lr = 0.02
    extra_params = {'batch_size':100}

# MOMENTUM GRADIENT DESCENT DEFAULTS
if args.optimizer == 'Momentum':
    epochs = 2
    lr = 0.02
    extra_params = {'gamma':0.9,'btch_sz':100}

# NESTEROV ACCELERATED DESCENT DEFAULTS
if args.optimizer == 'Nesterov':
    epochs = 10
    lr = 0.001
    extra_params = {'gamma':0.9, 'btch_sz':100}

# RMS PROP DESCENT DEFAULTS
if args.optimizer == 'RMSprop':
    epochs = 20
    lr = 0.001
    extra_params = {'beta':0.9, 'btch_sz':100, 'epsilon':1e-8}

# ADAGRAD DESCENT DEFAULTS
if args.optimizer == 'Adagrad':
    epochs = 35
    lr = 0.1
    extra_params = {'epsilon':1e-8, 'btch_sz':32}

# ADADELTA DESCENT DEFAULTS
if args.optimizer == 'Adadelta':
    epochs = 40
    lr = 0.001
    extra_params = {'beta':0.9, 'btch_sz':32, 'epsilon':1e-8}

# ADAM DESCENT DEFAULTS
if args.optimizer == 'Adam':
    epochs =1000
    lr = 0.01
    extra_params = {'beta1':0.9, 'beta2':0.99}

# NADAM DESCENT DEFAULTS
if args.optimizer == 'Nadam':
    epochs =1000
    lr = 0.01
    extra_params = {'beta1':0.9, 'beta2':0.99}

# ADAMAX DESCENT DEFAULTS
if args.optimizer == 'Adamax':
    epochs =1000
    lr = 0.01
    extra_params = {'beta1':0.9, 'beta2':0.99,'p':10e9}



# loading the dataset
cal = datasets.fetch_california_housing()
X = cal.data
y = cal.target
feature_names = cal.feature_names
print(f"Loaded {X.shape[0]} data points")

#Mean normalization
for j in range(X.shape[1]):
    X[:, j] = (X[:, j] - X[:,j].mean())/X[:, j].std()
print(f"Normalized the data ")

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X[:, 1], y, test_size = 0.2, shuffle=True, random_state = 0)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1,1)
print(f" Using {X_train.shape[0]} examples for training and {X_test.shape[0]} examples for testing.")
print(f" Shape of training data is {X_train.shape}")


print(f"Training")
final_theta, batch_history = training_fn(X = X_train, 
                                    y = y_train, 
                                    opt_fn = optimizing_fn,
                                    eps = epochs, 
                                    lr = lr, 
                                    extra_params = extra_params)

# TO TEST ON TEST SET
print(f"Test Mean squared Error = {find_mse(final_theta, X_test, y_test)}")

plot_cost_graph(batch_history['costs'][:2000])