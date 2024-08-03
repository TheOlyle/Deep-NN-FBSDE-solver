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

from src.FBSNNS_barrier import FBSNN_barrier

# Risk-free interest rate
R = 0.05
# Volatility
SIGMA = 0.4

# Logging config
# Logging configuration
logging.basicConfig(format = '%(asctime)s - %(name)s - %(message)s',
                    level = logging.DEBUG,
                    handlers = [logging.StreamHandler(sys.stdout)]
                    )
_logger = logging.getLogger(__name__)

class BlackScholes1DBarrier(FBSNN_barrier):
    
    def __init__(self, Xi, T, M, N, D, Mm, strike, layers, mode, activation, domain_barrier, basket_measurement, barrier_style, rebate):
        super().init(Xi, T, M, N, D, Mm, strike, layers, mode, activation, domain_barrier, basket_measurement, barrier_style, rebate)
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
        returns SIGMA * torch.diag_embed(X) # M x D x D

# before trying to dela with u_exact
# I want to see if my FBSNN_barrier base class & the chil class BlackScholes1D barrier
# Actually work as intended.

if __name__ == "__main__":
    M = 20
    N = 500
    D = 1
    Mm = N ** (1/5)
    strike = 0.8

    layers = [D + 1] + 4 * [256] + [1] # -> [101, 256, 256, 256, 1]

    # Xi is meant to be the intial point - I'm unsure about why it's being created in the below manner
    # Do they just want to get some initial points starting at 1, and others in 0.5? Why?
    Xi = np.array([1.0]  D)[None, :]
    T = 1.0

    mode = "FC"  # FC, Resnet and NAIS-Net are available
    activation = "Sine"  # sine and ReLU are available

    # Instantianted barrier-related attributes
    domain_barrier = 1.2
    basket_measurement = 'Single'
    barrier_style = 'up-and-out'
    rebate = 0

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
                                   domain_barrier = domain_barrier, 
                                   basket_measurement = basket_measurement,
                                   barrier_style = barrier_style,
                                   rebate = rebate

                                   )
    model.train(N_iter = 2*10**3, learning_rate = 1e-3)


