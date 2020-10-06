## This folder contains the code for testing out various optimizations described in the paper.
The following Algorithms are implemented here - 
1.  ### Batch Gradient Descent
    Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t.
    to the parameters θ for the entire training dataset

    Here's the cost graph obtained using The code implemented - 

    ![batch](./graphs/Batch.png)

2.  ### Stochastic Gradient Descent
    Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example

    Here's the cost graph obtained using the code implemented - 

    ![stochastic]()

3.  ### Mini Batch Gradient Descent
    Mini-batch gradient descent finally takes the best of both worlds and performs an update for every
    mini-batch of n training examples

    Here's the cost graph obtained using the code implemented - 

    ![minibatch]()

4.  ### Momentum
    Momentum [17] is a method that helps accelerate SGD in the relevant direction and dampens
    oscillations of the Gradient update. It does this by adding a fraction γ of the update vector of the
    past time step to the current update vector.

    Here's the cost graph obtained using the code implemented - 

    ![momentum]()

5.  ### Nesterov accelerated gradient
    Nesterov accelerated gradient (NAG) is a way to give our momentum term this kind of prescience.
    We know that we will use our momentum term γ vt−1 to move the parameters θ.

    Here's the cost graph obtained using the code implemented - 

    ![Nesterov]()

6.  ### Adagrad 
    Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters.

    Here's the cost graph obtained using the code implemented - 

    ![Adagrad]()

7.  ### Adadelta
    Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing
    learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of
    accumulated past gradients to some fixed size w.

    Here's the cost graph obtained using the code implemented - 

    ![Adadelta]()

8.  ### RMSprop
    RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.

    Here's the cost graph obtained using the code implemented - 

    ![rmsprop]()

9.  ### Adam
    Adaptive Moment Estimation (Adam) [10] is another method that computes adaptive learning rates
    for each parameter. In addition to storing an exponentially decaying average of past squared gradients vt like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients mt, similar to momentum.

    Here's the cost graph obtained using the code implemented - 

    ![adam]()

10. ### Adamax
    We generalize the gradient update rule of Adam to l<sub>p</sub> norm 

    Here's the cost graph obtained using the code implemented - 

    ![adamax]()

11. ### Nadam
    Nadam (Nesterov-accelerated Adaptive Moment Estimation) combines Adam and NAG. In
    order to incorporate NAG into Adam, we need to modify its momentum term mt.

    Here's the cost graph obtained using the code implemented - 

    ![nadam]()





   