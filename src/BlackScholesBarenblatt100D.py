# Config to allow debugger to work
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # This allows to run dwebugger with absolute imports

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from src.FBSNNs import FBSNN


class BlackScholesBarenblatt(FBSNN):
    """ 
    This class instantiates a child class, with Black-Scholes-Barenblatt
    as a SPECIAL case of the more general semi-linear PDE problem
    """
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  # M x D
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        return 0.4 * torch.diag_embed(X)  # M x D x D

    ###########################################################################


def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    """
    This returns the EXACT solution of the Black-Scholes-BarenBlatt equation
    with payoff function g(x) = x^2

    payoff function g(x) = x^2 is likely used because it's simple, allowing for
    a closed-form solution.
    """
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1


def run_model(model, N_Iter, learning_rate):
    tot = time.time()
    samples = 5
    print(model.device)
    graph = model.train(N_Iter, learning_rate)
    print("total time:", time.time() - tot, "s")

    np.random.seed(42)
    # Retriev the set of t and Brownian Motions
    t_test, W_test = model.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x 1
    # Retrieve the NN-predicted values given our simulation values of t & Brownian motion W
    X_pred, Y_pred = model.predict(Xi, t_test, W_test) # M x (N+1) x 1, M x (N+1) x 1

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()


    # Retrieve the EXACT Black-Scholes-B solution to test the output of the model against
    # This ends up being M x (N+1) x 1
    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])

    # Perform comparisons
    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()

    # Get the relative errors in positive terms
    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)


if __name__ == "__main__":
    tot = time.time()
    M = 50  # number of trajectories (batch size). Originally, M  = 100 
    N = 30  # number of time snapshots, Originally, M = 50
    D = 1  # number of dimensions. Originally, D = 100

    # Unclear why 1st layer is 101 instead of 100
    # But 256 neurons used in each hidden layer
    layers = [D + 1] + 4 * [256] + [1] # -> [101, 256, 256, 256, 1]

    # Xi is meant to be the intial point - I'm unsure about why it's being created in the below manner
    # Do they just want to get some initial points starting at 1, and others in 0.5? Why?
    # Note, I amended this to the standard for my own. Originally was Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    Xi = np.array([1.0] * D)[None, :]
    T = 1.0

    "Available architectures"
    mode = "FC"  # FC, Resnet and NAIS-Net are available
    activation = "Sine"  # sine and ReLU are available
    model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   layers, mode, activation)
    
    run_model(model, 1000, 1e-3)