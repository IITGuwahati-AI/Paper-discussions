import numpy as np

# BatchGradient descent
def Batch_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
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
            print(f"Completed {ep+1}/{epochs} epochs, with training error : {find_mse(theta, X, y)}")


    history = {
        'costs': cost_history,
        'errors': training_errors,
        'thetas': theta_history
    }
    return theta, history


def Stochastic_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
    # history storing
    cost_history = []
    training_errors = []

    for ep in range(epochs):

        print(f"\nStarting epoch {ep+1}")

        # shuffling the arrays
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X = X[ids,:]
        y = y[ids]

        for i in (range(X.shape[0])):
            grad = gradient_calc(theta, X[i, :].reshape(1,-1), y[i])
            theta = theta - (learning_rate * grad)

            if i%5000 == 0:
                print(f"Processed {i+1}/{X.shape[0]} examples, with training error : {find_mse(theta, X, y)} ")

            cost_history.append(cost_function(theta, X, y))
            training_errors.append(find_mse(theta, X, y))


    history = {
        'costs': cost_history,
        'errors': training_errors
    }
    return theta, history


def MiniBatch_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
    # history storing
    cost_history = []
    training_errors = []
    grad_history = []

    btch_sz = extra_params['batch_size']

    for ep in range(epochs):

        print(f"\nStarting epoch {ep+1}")

        # shuffling the arrays
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X = X[ids,:]
        y = y[ids]

        for i in range(0, X.shape[0], btch_sz):


            grad = gradient_calc(theta, X[i:i+btch_sz, :], y[i:i+btch_sz])
            theta = theta - (learning_rate * grad)
            
            grad_history.append(learning_rate * grad)
            
            if i%5000 == 0:
                print(f"Processed {i+1}/{X.shape[0]} examples, with training error : {find_mse(theta, X, y)} ")

            cost_history.append(cost_function(theta, X, y))
            training_errors.append(find_mse(theta, X, y))


    history = {
        'costs': cost_history,
        'errors': training_errors,
        'grads': grad_history
    }
    return theta, history