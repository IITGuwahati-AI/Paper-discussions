import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# UTILITY FUNCTIONS FOR IMPLEMENTATION
def cost_function(theta, X, y):
    m  = X.shape[0]
    X_with_ones = np.concatenate([np.ones((m, 1)), X], axis=1) # X shape (m, n+1)
    
    h = np.dot(X_with_ones,theta)
    J = (1/(2*m))*(np.sum(h-y)**2)

    return J

def gradient_calc(theta, X, y):
    m  = X.shape[0]
    X_with_ones = np.concatenate([np.ones((m, 1)), X], axis=1) # X shape (m, n+1)

    grad = (1/m)*np.transpose(X_with_ones)@(X_with_ones@theta - y)
    return grad

# for getting error
def find_mse(theta, X, y):
    m  = X.shape[0]
    X_with_ones = np.concatenate([np.ones((m, 1)), X], axis=1) # X shape (m, n+1)

    h = np.dot(X_with_ones,theta)
    return (mean_squared_error(y, h))

# function to train
def training_fn(X, y, opt_fn, eps, lr, extra_params = {}):
    theta = np.random.uniform(size = (X.shape[1]+1,))

    final_theta, hist = opt_fn(X = X,
                                y = y, 
                                theta = theta, 
                                epochs = eps, 
                                learning_rate = lr,
                                gradient_calc = gradient_calc,
                                cost_function= cost_function,
                                find_mse = find_mse, 
                                extra_params = extra_params)
    
    return final_theta, hist

def plot_cost_graph(costs):

    plt.plot(costs)
    plt.title('TRAINING COST GRAPH', fontsize=20, color='red')
    plt.xlabel('Epochs', fontsize=13, color='green')
    plt.ylabel('Costs', fontsize=13, color='green')
    plt.show()