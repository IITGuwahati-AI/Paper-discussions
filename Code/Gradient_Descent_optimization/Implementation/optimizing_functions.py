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