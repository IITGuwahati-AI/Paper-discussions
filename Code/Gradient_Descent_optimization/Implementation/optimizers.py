import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# loading the dataset
cal = datasets.fetch_california_housing()
X = cal.data
y = cal.target
feature_names = cal.feature_names
print(f"Loaded {X.shape[0]} data points ")

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
                                extra_params = extra_params)
    
    return final_theta, hist

def plot_cost_graph(costs):

    plt.plot(costs)
    plt.title('COST GRAPH\n', fontsize=20, color='white')
    plt.xlabel('Epochs', fontsize=13, color='white')
    plt.ylabel('Costs', fontsize=13, color='white')
    plt.show()

# BatchGradient descent
def BatchGD_optimization(X, y, theta, epochs, learning_rate, extra_params = {}):
    # history storing
    cost_history = []
    training_errors = []
    theta_history = []


    for ep in range(epochs):
        grad = gradient_calc(theta, X, y)
        theta = theta - (learning_rate * grad)

        cost_history.append(cost_function(theta, X, y))
        training_errors.append(find_mse(theta, X, y.reshape(-1,1)))
        theta_history.append(theta)

        if ep%1000 == 0:
            print(f"Completed {ep+1}/{epochs} epochs, with training error : {find_mse(theta, X, y)} and test error : {find_mse(theta, X_test, y_test)}")


    history = {
        'costs': cost_history,
        'errors': training_errors,
        'thetas': theta_history
    }
    return theta, history

print(f"Training")
final_theta, batch_history = training_fn(X = X_train, 
                                    y = y_train, 
                                    opt_fn = BatchGD_optimization,
                                    eps = 12000, 
                                    lr = 0.001, 
                                    extra_params = {})

# TO TEST ON TEST SET
print(f"MSE = {find_mse(final_theta, X_test, y_test)}")

plot_cost_graph(batch_history['costs'][:2000])