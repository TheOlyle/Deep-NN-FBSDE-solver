import sys
import numpy as np
from abc import ABC, abstractmethod
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.Models import *

from src.utils import (
    check_breach,
    update_C,
    update_tFP,
    update_XFP,
    update_YFP,
    NegRelu
)

# TensorBoard config: change for different experiments
tb_writer = SummaryWriter(log_dir = 'runs/Barrier_Call1D/K_1_Barr_100_learnrate_0p1')

# Logging configuration
logging.basicConfig(format = '%(asctime)s - %(name)s - %(message)s',
                    level = logging.DEBUG,
                    handlers = [logging.StreamHandler(sys.stdout)]
                    )
_logger = logging.getLogger(__name__)

class FBSNN_barrier(ABC):
    def __init__(self, Xi, T, M, N, D, Mm, strike, layers, mode, activation,
                 domain_barrier = None, basket_measurement = None, barrier_style =  None, rebate = None,
                 penalise_neg_Y = False, dynamic_lr = False # Miscellaneous parameters for how to action the NN.
                 ):
        # Constructor for the FBSNN class
        # Initializes the neural network with specified parameters and architecture
        
        # Parameters:
        # Xi: Initial condition (numpy array) for the stochastic process
        # T: Terminal time
        # M: Number of trajectories (batch size)
        # N: Number of time snapshots
        # D: Number of dimensions for the problem
        # Mm: Number of discretization points for the SDE # THIS IS USED IN PARPAS NEW WORK, NOT IN ORIGINAL! It's his way of actioning multi-level Monte Carlo I think
        # layers: List indicating the size of each layer in the neural network
        # mode: Specifies the architecture of the neural network (e.g., 'FC' for fully connected)
        # activation: Activation function to be used in the neural network

        # Check if CUDA is available and set the appropriate device (GPU or CPU)
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        # FROM ORIGINAL REPO: we set a random seed to ensure that your results are reproducible
        # torch.manual_seed(0)

        # Initialize the initial condition, convert it to a PyTorch tensor, and send to the device 
        self.Xi = torch.from_numpy(Xi).float().to(self.device)  # initial point -> this creates a Tensor
        self.Xi.requires_grad = True # Check why this needs requires_grad = True

        # Store other parameters as attributes of the class.
        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.Mm = Mm  # number of discretization points for the SDE
        self.strike = strike * self.D  # strike price - this is NOT in the original Repo. Unclear why he put this in (to simplify in case of vanilla payoff formula?)

        self.mode = mode  # architecture of the neural network
        self.activation = activation  # activation function        # Initialize the activation function based on the provided parameter
        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()
        elif activation == "Tanh":
            self.activation_function = nn.Tanh()

        # Specifying the barrier, if any
        self.domain_barrier = domain_barrier # Nonetype if no barrier, otherwise float. Indicates the RELATIVE barrier e.g. 1.7, relative to the intial fixing at X(t = 0)
        self.basket_measurement = basket_measurement # Nonetype if no barrier, otherwise one of ['WorstOf','BestOf','EqualWeighting']
        self.barrier_style = barrier_style # Nonteyp if no barrier, otherwise one of ['up-and-out', 'up-and-in']
        self.rebate = rebate # Nonetype if no barrier, otherwise float
        # Sense-checks
        if self.basket_measurement == 'Single': assert self.D == 1

        # Loss-modifying attributes
        self.penalise_neg_Y = penalise_neg_Y
        self.dynamic_lr = dynamic_lr

        # Initialize the neural network based on the chosen mode
        if self.mode == "FC":
            # Fully Connected architecture
            self.layers = []
            for i in range(len(layers) - 2):
                # For each layer, create linear transform e.g. Ax + b of the input
                # and then subsequently pass through an activation function
                # This de-facto achieves f(Ax+b)
                self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
                self.layers.append(self.activation_function)
            self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))

            # This connects the layers together within an nn.Sequential architecture - standard fully-connected approach to NN
            self.model = nn.Sequential(*self.layers).to(self.device)

        elif self.mode == "Naisnet":
            # NAIS-Net architecture
            self.model = Naisnet(layers, stable=True, activation=self.activation_function).to(self.device)

        # Apply a custom weights initialization to the model
        self.model.apply(self.weights_init)

        # Initialize lists to record training loss and iterations.
        self.training_loss = []
        self.iteration = []

        # Tensorboard: add graph here?

        _logger.DEBUG("finished initialising FBSNN_barrier")

    def weights_init(self, m):
        """
        Custom weight initialization method for neural network layers.
        However, it basically just wraps around the existing Xavier initialisation method
        Parameters:
        m: A layer of the neural network 
        """
        if type(m) == nn.Linear:
            # Initialize the weights of the linear layer using Xavier uniform initialization
            torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):  # M x 1, M x D
        # Computes the output of the neural network and its gradient with respect to the input state X
        # This is used when creating the loss function. From a given input, the NN gives its prediction of Y & Z (or u & dU, in general terms)
        
        # Parameters:
        # t: A batch of time instances, with dimensions M x 1
        # X: A batch of state variables, with dimensions M x D

        # Concatenate the time and state variables along second dimension
        # to form the input for the neural network
        input = torch.cat((t, X), 1)  

        # Pass the concatenated input through the neural network model
        # The output u is a tensor of dimensions M x 1, representing the value function at each input (t, X)
        u = self.model(input)  # M x 1 # This invokes the 'forward' method of our self.Model

        # Compute the gradient of the output u with respect to the state variables X
        # The gradient is calculated for each input in the batch, resulting in a tensor of dimensions M x D
        Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u), 
                                allow_unused=True, retain_graph=True, create_graph=True)[0]

        return u, Du

    def Dg_tf(self, tFP, XFP):  # M x D
        # Calculates the gradient of the function g with respect to the input X
        # THis is used when creating the loss function. This computes the gradient of the Payoff function with respect to X
        # This SHOULD be equivalent to our estimate for Z(T), the gradient between Y & X at time T,
        # since Y & X should have a deterministic relationship at that point with this calculable gradient
        # Y(T) = g(X(T))

        # Parameters:
        # X: A batch of state variables, with dimensions M x D

        # Because g_tf() is now dependent on tFP & XFP in a piece-wise manner,
        # NOTE: Big Problem - Dg_tf is giving a LOT of zero-values
        g = self.g_tf(tFP = tFP, XFP = XFP)  # M x 1

        # Now, compute the gradient of g with respect to X 
        # The gradient is calculated for each input in the batch, resulting in a tensor of dimensions M x D
        Dg = torch.autograd.grad(outputs=[g],
                                 inputs=[XFP],
                                 grad_outputs=torch.ones_like(g), 
                                 allow_unused=True,
                                 retain_graph=True,
                                 create_graph=True)[0] 

        return Dg


    def loss_function(self, t, W, Xi,
                      it: int # This is global training-epoch identifier for TensorBoard logs
                      C = None, tFP = None, XFP = None # These are all arguments exclusively for Barrier options
                      ):
        # Calculates the loss for the neural network
        # Recall that the Loss function used in Parpas' paper has 2 components
        #   1) Loss coming from NN-derived Y(t) not being internally consistent
        #   with the Euler-descretised Backward SDE
        #   2) Loss coming from NN-derived Y(T) not being equal to g(Euler-derived X(T)),
        #   as indicated in the terminal condition Y(T) = g(X(T))

        # Parameters:
        # t: A batch of time instances, with dimensions M x (N+1) x 1
        # W: A batch of Brownian motion increments, with dimensions M x (N+1) x D
        # Xi: Initial state, with dimensions 1 x D

        # Parameters for Barrier Options
        # C  (M x D Tensor) - Indicator variable if barrier has been breached at some point in this iteration & dimension. Takes place of XTrig from Ganesan et al
        # tFP (M x D Tensor) - Tracker variable to keep track of point-in-time at which barrier was breached, if any, in each iteration & dimension
        # XFP (M x D Tensor) - Tracker variable to keep track of value of X in each dimension at each iteration at which the barrier was breached, if it was indeed breached

        # Miscellaneous parameters
        # it (int) - identifies the epoch of training we are currently in. For TensorBoard dashboard/analytics.

        # Returns:
        #   (loss, ....)
        # Check out FBSNN.predict regarding this
        _logger.debug("Called loss_function")

        loss = 0  # Initialize the loss to zero.
        X_list = []  # List to store the states at each time step.
        Y_list = []  # List to store the network outputs at each time step.

        # Initial time and Brownian motion increment.
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        # Initial state for all trajectories
        # Xi is 1 x D Tensor (since all iterations start at same timepoint)
        # X0 is M X D Tensor, since we need to apply Euler-Discretisation logic across each dimension's realisation of W for each iteration
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # M x D

        # Obtain the network output and its gradient at the initial state: predicts (Y0, Z0) from (t0, X0)
        Y0, Z0 = self.net_u(t0, X0) # M x 1, M x D

        # Store the initial state and the network output
        X_list.append(X0)
        Y_list.append(Y0)
        
        # In the case of Barrier option, initialise YFP
        if self.domain_barrier:
            # I previously defined YFP = Y0.unsqueeze(2), getting an M x 1 x 1 tensor from an M x 1 tensor
            # However, on each iteration, we have a YFP value (which is shared across dimensions), which is initialised at Y0
            # This means we just need an M x 1 tensor, so same dimensions as Y0
            YFP = Y0
        
        # Iterate over each time step
        for n in range(0, self.N):
            # Next time step and Brownian motion increment
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            # Compute the next state using the Euler-Maruyama method - get X(t+1)
            # W1 - W0 will output M x D, so (W1 - W0).unsqueeze(-1) will output M x D x 1
            # This is likely to output M X D
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1) # torch.squeeze(..., dim = -1) is likely to remove the final dimension in the tensor (which is of size 1)
            
            # We get the NN's DIRECT prediction of Y(t+1), Z(t+1) based on inputs (t+1, X(t+1))
            Y1, Z1 = self.net_u(t1, X1)

            # Compute the Euler-descretised predicted value (Y1_tilde) at the next state - Y~(t+1)
            # We use the Neural-Nework predicted values for Y(t) & Z(t)
            # For barrier options, Y1_tilde follows different process in iterations where the barrer has been breached
            # Therefore, we update this timestep's Y1_tilde in the below if statement to check for barrier breaches
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1, keepdim=True) # We're summating, since this is what you do with high-dimensional PDE. Check the FInal Project brief

            # IF barrier is relevant, determine if the barrier has been hit ; update Y1_tilde for iterations where barrier has been hit
            if self.domain_barrier:
                # Get the performance of the X-dimensions across all iterations
                # Xi is of dimension 1 x D (starting point shared across all iterations), so need to 'repeat' it to be M x D and therefore do tensor algebra with X1, an M x D tensor
                perf = X1 / Xi.repeat(self.M, 1)

                # Make sure these arguments are correct!
                # Should I be use the pre-updated or post-update C/XTrig?
                # Should I be using t0 or t1 A: since i'm defining performance 'perf' based on X1, I should use t1
                tFP = update_tFP(tFP, C, t1)

                # Again, make sure these arguments are correct!
                # pre-updated or post-update C/XTrig? X1 or X0
                XFP = update_XFP(XFP, C, X1, self.D)

                # Again, make sure these arguments are correct! Y0 or Y1?
                # There's only one YFP for each iteration/path - M x 1 Tensor
                # Additionally, I need to make sure I incorporate logic from self.basket_measurement here
                # Just because a few of the X-dimensions breach the barrier, doesn't necessarily mean the overall Barrier Breach for Y necessarily - depends on self.basket_measurement
                # YFP is unused currently - Ganesan et al uses it in the loss function, why don't we?
                YFP = update_YFP(YFP, C, Y1)

                # Have moved C here - this makes the behaviour of tracker variables tFP, XFP & YFP behave accordingly - they record the values at the appropriate timestep for barrier breaches
                C = update_C(C = C,
                             performance=perf,
                             barrier = self.domain_barrier, 
                             measurement=self.basket_measurement,
                             style = self.barrier_style
                             )

                # Replace the Y1_tilde created earlier (before barrier if-statement) with one which
                # takes into account if the option's barrier has been breached
                # NOTE: the below implementation is ONLY for 'up-and-out' barrier options
                Y1_tilde = torch.where(C == 0,
                                       Y1_tilde, # meaning: if no barrier breach in this iteration, maintain same value as previously calculated
                                       self.rebate # meaining: if there IS a barrier breach in this iteration, follow the rebate value
                                       )

            # Add the squared difference between Y1 and Y1_tilde to the loss
            # This is loss derived from the difference in the NN's predicted Y(t+1) & the Euler-discretised estimate of Y(t+1)
            # The below should be dependent on if the barrier has been hit
            # Since we adjust the process followed by Y1_tilde on paths where the barrier has been hit, the below is de-factor dependent
            # on a barrier breach
            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            # NOTE: can also include some measure of loss derived from 'in-out' Parity OR using Yu et al's more mathematical approach?
            # To be determined if this loss should occur at each time-step or not, only at the very end. It makes sense to make this for each timestep
            # For now, I'm not going to implement this - need to make sure I can get Ganesan et al's methodology to work, and can then extend
            loss += 0

            # Sometimes, the NN's prediction of Y1 can be negative. Therefore, we couuld try to speed up convergence by introducing
            # strong penalties in the loss function for negative values of Y1?
            if self.penalise_neg_Y:
                loss += NegRelu(Y1)

            # Update the variables for the next iteration - we reset our values
            # so we can pass onto next timestep fresh
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1

            # Store the current state and the network output
            X_list.append(X0)
            Y_list.append(Y0)

        _logger.debug("Finished iterating over all timesteps - computing final loss component of terminal condition")         
        
        # After having computed all the losses arising from the 1st term, we now compute the loss
        # arising from the difference in NN-predicted value of Y(T) & the Euler-derived value of X(T)
        # wrapped around the terminal condiiton function g(.). 
        # NOTE: the below should be dependent on if a barrier has been hit - Ganesan et al use (YFP - g_tf(tFP, XFP))^2, but not sure how to implement this here ...
        loss += torch.sum(torch.pow(Y1 - self.g_tf(tFP = tFP, XFP = XFP)))

        #  We also include the loss from NN's predicted value of Z(T) (the gradient between Y & X at time T)
        #  being different from the gradient of the function g(.) on the Euler-derived value of X(T)
        #  Add the difference between the network's gradient and the gradient of g at the final state
        #  NOTE: the below should be dependent on if a barrier has been hit - Ganesan et al u
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(tFP = tFP, XFP = XFP), 2))

        # At THIS point, I should log g_tf() & Dg_tf() in TensorBoard. May as well also log Y1 & Z1
        # However, to do this, I need a unique identifier for this epoch: I should pass 'it' as an argument to loss function
        if it % 100 == 0:
            tb_writer.add_histogram(tag = 'Dg_tf at T',
                                    values = self.Dg_tf(tFP = tFP, XFP = XFP),
                                    global_step = it
                                    )
            tb_writer.add_histogram(tag = 'g_tf at T',
                                    values = self.g_tf(tFP = tFP, XFP = XFP),
                                    global_step = it
                                    )
            tb_writer.add_histogram(tag = 'Y_pred at T',
                                    values = Y1,
                                    global_step = it
                                    )
            tb_writer.close()


        # Create a list of Euler-derived X values & list of NN-derived Y-values
        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        # Return the loss and the states and outputs at each time step
        # This is Loss, Euler-X & NN-Y values, and
        # The final element returned is the first element of the network output, for reference or further use
        return loss, X, Y, Y[0, 0, 0]


    def fetch_minibatch(self):  # Generate time + a Brownian motion
        # Generates a minibatch of time steps and corresponding Brownian motion paths
        # This is used at the training stage. We split data into minibatches & subsequently train them
        # in downstream function.
        # It makes sense to think of minibatches as different realisations of Brownian Motion
        # Returns:
        #   overall entire set of paths of t and Brownian Motion W.

        T = self.T  # Terminal time
        M = self.M  # Number of trajectories (batch size)
        N = self.N  # Number of time snapshots
        D = self.D  # Number of dimensions

        # Initialize arrays for time steps and Brownian increments
        Dt = np.zeros((M, N + 1, 1))  # Time step sizes for each trajectory and time snapshot
        DW = np.zeros((M, N + 1, D))  # Brownian increments for each trajectory, time snapshot, and dimension

        # Calculate the time step size
        dt = T / N

        # Populate the time step sizes for each trajectory and time snapshot (excluding the initial time)
        Dt[:, 1:, :] = dt

        # Generate Brownian increments for each trajectory and time snapshot
        DW_uncorrelated = np.sqrt(dt) * np.random.normal(size=(M, N, D))
        DW[:, 1:, :] = DW_uncorrelated # np.einsum('ij,mnj->mni', self.L, DW_uncorrelated) # Apply Cholesky matrix to introduce correlations

        # Cumulatively sum the time steps and Brownian increments to get the actual time values and Brownian paths
        t = np.cumsum(Dt, axis=1)  # Cumulative time for each trajectory and time snapshot
        W = np.cumsum(DW, axis=1)  # Cumulative Brownian motion for each trajectory, time snapshot, and dimension. 

        # Convert the numpy arrays to PyTorch tensors and transfer them to the configured device (CPU or GPU)
        t = torch.from_numpy(t).float().to(self.device) # M x (N+1) x 1 # Remember: self.N (num of time snapshots) is affected by Multi-Levelling of Monte Carlo (self.N is adjusted each training loop)
        W = torch.from_numpy(W).float().to(self.device) # M x (N+1) x D # Remember: self.N (num of time snapshots) is affected by Multi-Levelling of Monte Carlo (self.N is adjusted each training loop)

        # Logic for if there is a Barrier:
        if self.domain_barrier:
            Xi = self.Xi # 1 x D - This is the initial point. The same idmension in each iteration starts at the same point

            C = torch.zeros((M, 1)) # For each iteration and each dimension, there is an XTrig value which initialises at 0
            tFP = torch.zeroes((M, 1)) # for each iteration and each dimension, there is a tFP value which initialises at same value as the 1st timestep of that iteration and dimension (which is 0 as welll)

            # For each iteration and each dimenison, there is an XFP value which initialises at the same level as the Xi for that dimension
            # We use repeat here, because we want to repeat the initial point across all iterations (even though it's the same anyway)
            # This is because subsequently, each iteration (and each of its dimensions) will deviate, and so we need to keep track of XFP across all the iterations & dimensions as well
            XFP = Xi.repeat(M, 1) # M x D, since X1 starts off being M x 1

            # NOTE: YFP is NOT defined here - as we only get access to Y0 at the start of the FBSNN.loss_function

            return t, W, C, tFP, XFP

        # Return the time values and Brownian paths.
        # If barrier is not active, we'll return None in place of C, tFP, XFP
        return t, W, None, None, None

    def train(self, N_Iter, learning_rate):
        # Train the neural network model.
        # Parameters:
        # N_Iter: Number of iterations for the training process
        # learning_rate: Learning rate for the optimizer
        _logger.debug("Started training")
        # Initialize an array to store temporary loss values for averaging
        loss_temp = np.array([])

        # Retrieve the current epoch/iteration of training we are on
        # Check if there are previous iterations and set the starting iteration number
        previous_it = 0
        if self.iteration != []:
            previous_it = self.iteration[-1]

        # Set up the optimizer (Adam) for the neural network with the specified learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Record the start time for timing the training process
        start_time = time.time()

        # Training loop
        for it in range(previous_it, previous_it + N_Iter): # Smart way of controlling how many training epochs to run each time from most recently trained model onward
            _logger.debug("started training loop %s", it)
            if it >= 4000 and it < 20000: # This is his way of using Multi-Level Monte Carlo I guess?
                # Updates self.N (num of time snapshots) to rounded-up value of self.Mm scaled by the value of whichever iteration we are of the first 4000th
                self.N = int(np.ceil(self.Mm ** (int(it / 4000) + 1)))
            elif it < 4000:
                self.N = int(np.ceil(self.Mm))

            # It is important to reset gradients of each parameter to 0 at the start of the iteration
            # hece, zero_grad()
            self.optimizer.zero_grad()

            # Retrieve the 'mini-batches' of time steps & Brownian motion paths - re-retrieved each time the model goes through the training epoch
            # C, tFP, XFP are only applicable for barrier options
            # We then compute the loss for the current batch
            if self.domain_barrier:
                t_batch, W_batch, C_batch, tFP_batch, XFP_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D, M x D, M x D, M x 1 x D

                loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi,
                                                                   it,
                                                                   C_batch, tFP_batch, XFP_batch)
            
            else:
                t_batch, W_batch, _ = self.fetch_minibatch()                
                loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi, it)

            
            # Perform backpropagation - computes gradients of Loss tensore wrt to parameters using back-prop
            self.optimizer.zero_grad()  # Zero the gradients again to ensure correct gradient accumulation
            loss.backward()  # Compute the gradients of the loss w.r.t. the network parameters
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # TensorBoard: log the gradients
            if it % 100 == 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Add a Hook here to debug issues with gradients.
                        tb_writer.add_histogram(f'{name}_grad', param.grad, it)
                tb_writer.close()
            self.optimizer.step()  # Update the network parameters based on the gradients

            # Store the current loss value for later averaging - loss_temp is our store of the loss
            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            # Print the training progress every 100 iterations
            if it % 100 == 0:
                elapsed = time.time() - start_time  # Calculate the elapsed time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                    (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()  # Reset the start time for the next print interval

            # Record the average loss and iteration number every 100 iterations in
            # self.training_loss
            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())  # Append the average loss
                tb_writer.add_scalar(tag = 'Running loss',
                                     scalar_value = loss_temp.mean(),
                                     global_step = it
                                     )
                tb_writer.close()
                loss_temp = np.array([])  # Reset the temporary loss array
                self.iteration.append(it)  # Append the current iteration number

            # Log to TensorBoard:
            if it % 100 == 0:
                # Log Y0_pred
                tb_writer.add_scalar(tag = 'Y0_pred',
                                     scalar_value = Y0_pred,
                                     global_step = it
                                     )
                tb_writer.add_histogram(tag = 'All_Y_pred',
                                        values = Y_pred,
                                        global_step = it
                                        )
                tb_writer.close()

        tb_writer.flush() # good practice to flush at end of training.
        
        _logger.debug("finished training")
        

        # Stack the iteration and training loss for plotting
        graph = np.stack((self.iteration, self.training_loss))

        # Return the training history (iterations and corresponding losses)
        return graph

    def predict(self, Xi_star, t_star, W_star):
        # Predicts the output of the neural network
        # Parameters:
        # Xi_star: The initial state for the prediction, given as a numpy array
        # t_star: The time steps at which predictions are to be made
        # W_star: The Brownian motion paths corresponding to the time steps

        # Convert the initial state (Xi_star) from a numpy array to a PyTorch tensor
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True

        # Compute the loss and obtain predicted states (X_star) and outputs (Y_star) using the trained model
        loss, X_star, Y_star, Y0_pred = self.loss_function(t_star, W_star, Xi_star)

        # Return the predicted states and outputs
        # These predictions correspond to the neural network's estimation of the state and output at each time step
        return X_star, Y_star

    def save_model(self, file_name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_loss': self.training_loss,
            'iteration': self.iteration
        }, file_name)
    
    def load_model(self, file_name):
        checkpoint = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_loss = checkpoint['training_loss']
        self.iteration = checkpoint['iteration']

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        # Abstract method for defining the drift term in the SDE
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Z: Gradient of the function with respect to state variables, size M x D
        # Expected return size: M x 1
        pass

    @abstractmethod
    def g_tf(self, X):  # M x D
        # Abstract method for defining the terminal condition of the SDE
        # Parameter:
        # X: Terminal state variables, size M x D
        # Expected return size: M x 1
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        # Abstract method for defining the drift coefficient of the underlying stochastic process
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Z: Gradient of the function with respect to state variables, size M x D
        # Default implementation returns a zero tensor of size M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        # Abstract method for defining the diffusion coefficient of the underlying stochastic process
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Default implementation returns a diagonal matrix of ones of size M x D x D
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D
