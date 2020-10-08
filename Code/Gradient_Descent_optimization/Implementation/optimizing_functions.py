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

# STOCHASTIC
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

# MINI BATCH
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

# MOMENTUM
def Momentum_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
    # history storing
    cost_history = []
    training_errors = []
    grad_history = []

    gamma = extra_params['gamma']
    btch_sz = extra_params['btch_sz']

    for ep in range(epochs):

        print(f"\nStarting epoch {ep+1}")

        # shuffling the arrays
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X = X[ids,:]
        y = y[ids]

        v_t = 0

        for i in range(0, X.shape[0], btch_sz):


            grad = gradient_calc(theta, X[i:i+btch_sz, :], y[i:i+btch_sz])
            v_t = (gamma * v_t) + (learning_rate * grad)
            theta = theta - v_t
            
            grad_history.append(v_t)

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

# NESTEROV
def Nesterov_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
    # history storing
    cost_history = []
    training_errors = []

    gamma = extra_params['gamma']
    btch_sz = extra_params['btch_sz']

    for ep in range(epochs):

        print(f"\nStarting epoch {ep+1}")

        # shuffling the arrays
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X = X[ids,:]
        y = y[ids]

        v_t = 0

        for i in range(0, X.shape[0], btch_sz):


            grad = gradient_calc(theta - (gamma* v_t), X[i:i+btch_sz, :], y[i:i+btch_sz])
            v_t = (gamma * v_t) + ((learning_rate * grad))
            theta = theta - v_t

            if i%5000 == 0:
                print(f"Processed {i+1}/{X.shape[0]} examples, with training error : {find_mse(theta, X, y)} ")

            cost_history.append(cost_function(theta, X, y))
            training_errors.append(find_mse(theta, X, y))

    history = {
        'costs': cost_history,
        'errors': training_errors
    }
    return theta, history

# RMS PROP
def RMSprop_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
    # history storing
    cost_history = []
    training_errors = []

    beta = extra_params['beta']
    btch_sz = extra_params['btch_sz']
    epsilon = extra_params['epsilon']

    for ep in range(epochs):

        print(f"\nStarting epoch {ep+1}")

        # shuffling the arrays
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X = X[ids,:]
        y = y[ids]

        v_t = 0

        for i in range(0, X.shape[0], btch_sz):


            grad = gradient_calc(theta , X[i:i+btch_sz, :], y[i:i+btch_sz])
            v_t = (beta * v_t) + (1-beta)*np.square(grad)
            theta = theta - ((learning_rate/(np.sqrt(np.square(v_t)+epsilon))) * grad)

            if i%5000 == 0:
                print(f"Processed {i+1}/{X.shape[0]} examples, with training error : {find_mse(theta, X, y)} ")

            cost_history.append(cost_function(theta, X, y))
            training_errors.append(find_mse(theta, X, y))


    history = {
        'costs': cost_history,
        'errors': training_errors
    }
    return theta, history

# ADAGRAD
def Adagrad_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
    
    cost_history = []
    training_errors = []

    no_of_params = theta.shape[0]
    epsilon = extra_params['epsilon']
    btch_sz = extra_params['btch_sz']
    

    Gt_ii=0

    for ep in range(epochs):

        print(f"\nStarting epoch {ep+1}")

        # shuffling the arrays
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X = X[ids,:]
        y = y[ids]

        for i in range(0, X.shape[0], btch_sz):

            g_ti = gradient_calc(theta , X[i:i+btch_sz, :], y[i:i+btch_sz])

            Gt_ii += g_ti**2

            theta = theta - (learning_rate/(np.sqrt(Gt_ii+epsilon)))*g_ti

        cost_history.append(cost_function(theta, X, y))
        training_errors.append(find_mse(theta, X, y.reshape(-1,1)))
        
        print(f"Completed {ep+1}/{epochs} epochs, with training error : {find_mse(theta, X, y)} ")

    history = {
            'costs': cost_history,
            'errors': training_errors
        }

    return theta, history

# ADADELTA
def Adadelta_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):
    # history storing
    cost_history = []
    training_errors = []

    beta = extra_params['beta']
    btch_sz = extra_params['btch_sz']
    epsilon = extra_params['epsilon']


    for ep in range(epochs):

        print(f"\n Running epoch {ep+1}")

        # shuffling the arrays
        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X = X[ids,:]
        y = y[ids]

        v_t = 0

        for i in range(0, X.shape[0], btch_sz):


            grad = gradient_calc(theta , X[i:i+btch_sz, :], y[i:i+btch_sz])
            v_t = (beta * v_t) + (1-beta)*np.square(grad)
            theta = theta - ((learning_rate/(np.sqrt(np.square(v_t)+epsilon))) * grad)

            cost_history.append(cost_function(theta, X, y))
            training_errors.append(find_mse(theta, X, y))


    history = {
        'costs': cost_history,
        'errors': training_errors
    }
    return theta, history

# ADAM algorithm
def Adam_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):

    cost_history = []
    training_errors = []

    beta1 = extra_params['beta1']
    beta2 = extra_params['beta2']
    lr = learning_rate
    epsilon = 1e-8
    mt = np.zeros(theta.shape)
    vt = 0.
    mt_unbiased = np.zeros(theta.shape)
    vt_unbiased = 0.

    for ep in range(epochs):

        gr = gradient_calc(theta,X,y)
        mt = mt*beta1 + (1-beta1)*gr
        vt = vt*beta2 + (1-beta2)*(np.sum(gr*gr))
        mt_unbiased = mt/(1-beta1**(ep+1))
        vt_unbiased = vt/(1-beta2**(ep+1))
        theta = theta -((lr/(np.sqrt(vt_unbiased)+epsilon))*mt_unbiased)

        cost_history.append(cost_function(theta,X,y))

        if((ep+1)%100 == 0):
            print( ep+1, cost_function(theta,X,y))

    history = {
        'costs': cost_history,
        'errors': training_errors
    }
    
    return theta, history

# NADAM algorithm
def Nadam_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):

    cost_history = []
    training_errors = []

    beta1 = extra_params['beta1']
    beta2 = extra_params['beta2']
    lr = learning_rate
    epsilon = 1e-8
    mt = np.zeros(theta.shape)
    vt = 0.
    mt_unbiased = np.zeros(theta.shape)
    vt_unbiased = 0.

    for ep in range(epochs):

        gr = gradient_calc(theta,X,y)
        mt = mt*beta1 + (1-beta1)*gr
        vt = vt*beta2 + (1-beta2)*(np.sum(gr*gr))
        mt_unbiased = mt/(1-beta1**(ep+1))
        vt_unbiased = vt/(1-beta2**(ep+1))
        theta = theta -((lr/(np.sqrt(vt_unbiased)+epsilon))*(beta1*mt_unbiased + (1-beta1)*gr/(1-beta1**(ep+1))))

        cost_history.append(cost_function(theta,X,y))

        if((ep+1)%100 == 0):
            print( ep+1, cost_function(theta,X,y))

    history = {
        'costs': cost_history,
        'errors': training_errors
    }
    
    return theta, history

# ADAMAX algorithm
def Adamax_optimization(X, y, theta, epochs, learning_rate, gradient_calc, cost_function, find_mse, extra_params = {}):

    cost_history = []
    training_errors = []

    beta1 = extra_params['beta1']
    beta2 = extra_params['beta2']
    p = extra_params['p']
    lr = learning_rate
    epsilon = 1e-8
    mt = np.zeros(theta.shape)
    vt = 0.
    mt_unbiased = np.zeros(theta.shape)
    vt_unbiased = 0.

    for ep in range(epochs):

        gr = gradient_calc(theta,X,y)
        mt = mt*beta1 + (1-beta1)*gr
        if(p<10**8):
            vt = vt*(beta2**p) + (1-(beta2**p))*((np.sum(gr*gr))**(p/2))
        else:
            vt = np.maximum(beta2*vt,np.sqrt(np.sum(gr*gr)))
        
        mt_unbiased = mt/(1-beta1**(ep+1))

        theta = theta -(lr/vt)*mt_unbiased

        cost_history.append(cost_function(theta,X,y))

        if((ep+1)%100 == 0):
            print( ep+1, cost_function(theta,X,y))

    history = {
        'costs': cost_history,
        'errors': training_errors
    }
    
    return theta, history

