""" 
This is tfor the example of a 1-D Barrier Call option
this is the simplest case
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # This allows to run dwebugger with absolute imports

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging
from math import log, exp
from scipy.stats import norm
import json

from torch.distributions.normal import Normal

from src.FBSNNS_barrier import FBSNN_barrier

# Risk-free interest rate
R = 0.04
# Volatility
SIGMA = 0.2

# Logging config
# Logging configuration
logging.basicConfig(format = '%(asctime)s - %(name)s - %(message)s',
                    level = logging.DEBUG,
                    handlers = [logging.StreamHandler(sys.stdout)]
                    )
_logger = logging.getLogger(__name__)

class BlackScholes1DBarrier(FBSNN_barrier):
    
    def __init__(self, Xi, T, M, N, D, Mm, strike, layers, mode, activation, N_Iter, learning_rate, tb_log, domain_barrier, basket_measurement, barrier_style, rebate, penalise_neg_Y):
        super().init(Xi, T, M, N, D, Mm, strike, layers, mode, activation, N_Iter, learning_rate, tb_log,
                     domain_barrier, basket_measurement, barrier_style, rebate,
                     penalise_neg_Y)
        # We are restricting ourselves to the special case of 1-underlyig Barrier optino
        self.basket_measurement = 'Single'
        self.D = 1

    def phi_tf(self, t, X, Y, Z):
        """
        This is drift term in the SDE for Y
        Per Parpas lecture, this is r * Y
        BUT
        looking at mu_tf, he initialises that as an M x D zero-tensor. Why not
        this coefficient phi_tf?
        On the other hand, in CallOption(FBSNN), Parpas just sets phi_tf = R * (Y) => i'll use this
        """
        return R * (Y) # same dim as Y, M x 1
    
    def g_tf(self, tFP, XFP):
        """
        This is the terminal condition for the SDE
        Given this is a Barrier option, this is actually a piece-wise function
        Read through Hull & Ganesan et al before defining this

        SO essentially:
            This needs to action Ganesan et al's formulation of g_B(t, X) => g_B(tFP, XFP)
            IF t = T, then tFP = T and so barrier has not been hit
            and so payoff is either standard call (up-and-out-call) or rebate (up-and-in)
            IF t < T, then tFP < T, and so barrier has been hit
            and so payoff is either rebate (up-and-out call) or Value of vanilla Call/Payoff of vanilla call for up-and-in call (Ganesan et al is weird here - need to research this further)
        
        Args:
            XFP (M x D = M x 1) - we on only have 1 dimension/X-process to keep track of here
            tFP (M x 1) - we only have 1 dimensions to keep track of here

        Returns:
            we need to return an M x 1 tensor - payoff of each path
        """
        T = self.T
        M = self.M
        D = self.D
        strike = self.strike

        # Define a standard call payoff
        call_payoff = lambda x,k : torch.maximum(x - k, torch.zeros((M, D))) # x is XFP (M x D = M x 1), K is a constant
        strike_tensor = torch.full((M, D), strike)

        if self.barrier_style == 'up-and-out':
            temp = torch.where(tFP == T, # in iterations where barrier has NOT been hit
                               call_payoff(XFP, strike_tensor), # in elements when tFP == T, then XFP = Terminal X(T)
                               self.rebate
                               )
            
        elif self.barrier_style == 'up-and-in':
            temp = torch.where(tFP == T, # In iterations where barrier has NOT been hit
                               self.rebate, 
                               call_payoff(XFP, strike_tensor) # Check on this - this contradicts Ganesan et al, but matches my understanding of up-and-in options
                               )

    def mu_tf(self, t, X, Y, Z):
        """
        This is the drift term for the SDE in X
        In Black-Scholes, this is r * X
        HOWEVER - Parpas impelemntation of this implements the abstract method mu_tf, 
        which initialisees an M x D zero-tensor. Why? This gets updated every time?
        Not sure what he's trying to do there

        BUT in Parpas' implementation of CallOption, he does it normally? As in R * X => using this
        """
        return R * (X) # Same dim as X, M x D
    
    def sigma_tf(self, t, X, Y):
        """
        This is the diffisuion coefficent for the SDE in X AND in Y
        FOr the X-process in Black Scholes: this is sigma * X ( * dW)
        For the Y-process in Black-Scholes: this is sigma * X ( * Z * dW)
        -> the 'Z' term is taken care of at the point of computation in FBSNN.loss_function
        
        Parpas uses torch.diag_embed.
        This converts X, an M x D tensor, into an M x D x D tensor
        Essentially, returns a batch D x D diagonal matrices for the diffusion coefficients
        """
        return SIGMA * torch.diag_embed(X) # M x D x D

# before trying to dela with u_exact
# I want to see if my FBSNN_barrier base class & the chil class BlackScholes1D barrier
# Actually work as intended.

def torch_vanilla_call(t:torch.Tensor, X:torch.Tensor, T, K, r, vol) -> torch.Tensor:
    """
    Using this to get vanilla call price, as this is used to get value of up-and-out barrier call price.
    has been checked
    """
    # Utility function, normal cumulative distribution
    standard_nrom = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ncdf = lambda tensor: standard_norm.cdf(tensor)

    # Getting base call price
    ## d1
    d1_numer = torch.lox(X/K) + (r + (vol**2)/2) * (T-t)
    di_denom = vol * (T - t)**0.5

    d1 = d1_numer/d1_denom

    d2 = d1 - d1_denom

    vanilla_call = X * ncdf(d1) - K * torch.exp(-r * (T-t)) * ncdf(d2)

    return vanilla_call

def u_exact(t:torch.Tensor, X:torch.Tensor, T, K, r, vol, H, barrier_style):
    """
    this actions analytical barrier solution for 1D problems, for tensors
    t: (N+1) x 1
    X: (N+1) x 1 

    this has been verified against QuantLib  
    """
    _logger.debug(f"args are {json.dumps(locals(), indent = 4, default = str)}")
    # Utility function: normal cumulative distribution
    standard_nrom = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    ncdf = lambda tensor: standard_norm.cdf(tensor)

    # Preliminaries: lam (lambda) and Y in Hull's formulation
    lam = (r + (vol**2)/2) / vol**2
    y = torch.log(H**2 / (X*K) ) / (vol * (T - t)**0.5) + lam * vol * (T - t)**0.5
    x1 = torch.log(X/H) / (vol * (T - t)**0.5) + lam * vol * (T - t)**0.5
    y1 = torch.log(H/X) / (vol * (T - t)**0.5) + lam * vol * (T - t)**0.5

    # Value of up-and-in
    term1 = X * ncdf(x1)
    term2 = K * torch.exp(-r * (T - t)) * ncdf(x1 - vol*(T - t)**0.5)
    term3 = - X * (H/K)**(2*lam) * ( ncdf(-y) - ncdf(-y1))
    term4 = K * torch.exp(-r * (T - t)) * (H/X)**(2*lam -2) * ( ncdf(-y + vol*(T - t)**0.5) - ncdf(-y1 + vol * (T - t)**0.5))

    up_in = term1 + term2 + term3 + term4

    vanilla_call = torch_vanilla_call(t = t, X = X, T = T, K = K, r = r, vol = vol)

    up_out = vanilla_call - up_in

    if barrier_style == 'up-and-out':
        return up_out
    elif barrier_style == 'up-and-in':
        return up_in

def graph(model, iteration, training_loss, t_test, Y_test, Y_pred, mean_errors, std_errors):
    """
    Function to handle graphics creation
    """
    sample = 5
    # Perform comparions
    plt.figure()
    plt.plot(iteration, training_loss) # previously graph[0] and graph[1]
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')

    # Convert tensor inputs into numpy arrays, for us in matplotlib
    t_test = t_test.detach().numpy()
    Y_test = Y_test.detach().numpy()
    Y_pred = Y_pred.detach().numpy()
    mean_errors = mean_errors.detach().numpy()
    std_errors = std_errors.detach().numpy()

    plt.figure()
    plt.plot(t_test[0:1, :].T, Y_pred[0:1, :].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :].T, Y_test[0:1, :].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1], Y_test[0:1, -1], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :].T, Y_pred[1:samples, :].T, 'b')
    plt.plot(t_test[1:samples, :].T, Y_test[1:samples, :].T, 'r--')
    plt.plot(t_test[1:samples, -1], Y_test[1:samples, -1], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()

    plt.figure()
    plt.plot(t_test[0, :], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)

    return

def run_model(model, N_Iter, learning_rate):
    """
    My version of run_model from BlackScholesBarenblatt100D
    """
    tot = time.time()

    # Train the model AND retrieve the (iteration, loss) numpy as stack at the end
    it_loss_stack = model.train(N_Iter, learning_rate)
    print("total time:", time.time() - tot, "s")

    np.random.seed(42)

    # Retriev the set of t and Brownian Motions
    t_test, W_test, C, tFP, XFP = model.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x 1
    # Retrieve the NN-predicted values given our simulation values of t & Brownian motion W
    X_pred, Y_pred = model.predict(Xi, t_test, W_test, C, tFP, XFP) # M x (N+1) x 1, M x (N+1) x 1

    # Reshape the tensors
    squeezed_t_test = t_test.squeeze(-1)
    squeezed_X_pred = X_pred.squeeze(-1)
    squeezed_Y_pred = Y_pred.squeeze(-1)

    # Create Y test from the exact available solutions
    Y_test = u_exact(t = squeezed_t_test, 
                     X=squeezed_X_pred,
                     T = T,
                     K = strike,
                     r = R,
                     vol = SIGMA,
                     H = domain_barrier, 
                     barrier_style=barrier_style
                     )
    # Calculate errors:
    # I'm not sure how this is mean to work: some paths end up being OTM,
    # which means Y_test would be 0 at t = T, and errors want to divide by 0
    # How did it work for Parpas? A: his boundary solution won't give 0 ever (unless X = 0)
    errors = torch.sqrt((Y_test - squeezed_Y_pred) ** 2 / Y_test ** 2)
    mean_errors = torch.mean(errors, 0)
    std_errors = torch.std(errors, 0)

    # Log in TensorBoard:
    # ...

    # Generate graph:
    graph(model = model,
          iteration = it_loss_stack[0],
          training_loss = it_loss_stack[1],
          t_test = squeezed_t_test, # M x (N+1)
          Y_test = Y_test, # M x (N+1)
          Y_pred = squeezed_Y_pred, # M x (N+1)
          mean_errors= mean_errors,
          std_errors= std_errors
          )

    return


if __name__ == "__main__":
    M = 50
    N = 30
    D = 1
    Mm = N ** (1/5)
    strike = 1

    layers = [D + 1] + 4 * [256] + [1] # -> [101, 256, 256, 256, 1]

    # Xi is meant to be the intial point - I'm unsure about why it's being created in the below manner
    # Do they just want to get some initial points starting at 1, and others in 0.5? Why?
    Xi = np.array([1.0] * D)[None, :]
    T = 1.0

    mode = "FC"  # FC, Resnet and NAIS-Net are available
    activation = "Sine"  # sine and ReLU are available

    # Instantianted barrier-related attributes
    domain_barrier = 10000 # Setting this at 100 makes this de-facto a standard call
    basket_measurement = 'Single'
    barrier_style = 'up-and-out'
    rebate = 0

    # Training Hyperparameters
    N_Iter = 12000
    learning_rate = 0.01 # Either 'dynamic' or a float

    model = BlackScholes1DBarrier(Xi = Xi,
                                    T = T,
                                    M = M,
                                    N = N,
                                    D = D,
                                    Mm = Mm,
                                    strike = strike,
                                    layers = layers,
                                    mode = mode,
                                    activation = activation,
                                    N_Iter= N_Iter,
                                    learning_rate=learning_rate
                                    domain_barrier = domain_barrier, 
                                    basket_measurement = basket_measurement,
                                    barrier_style = barrier_style,
                                    rebate = rebate,
                                    penalise_neg_Y = True,
                                    N_Iter = N_Iter,
                                    learning_rate = learning_rate,
                                    tb_log = "runs/Barrier_Call_1D/dev3"
                                    )
    
    run_model(model, 2000, 0.01)

