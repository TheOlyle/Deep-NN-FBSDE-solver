import sys
import os
sys.path.append(os.path.abspath("Algorithms/"))
sys.path.append(os.path.abspath("models/"))
#%%
from FBSNNs import *
from CallOption import *
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

M = 1 # number of trajectories (batch size)
N = 50  # number of time snapshots
D = 1 # number of dimensions
Mm = N ** (1/5)

layers = [D + 1] + 4 * [256] + [1]

Xi = np.array([1.0] * D)[None, :]
T = 1.0

"Available architectures"
mode = "Naisnet"  # FC and Naisnet are available
activation = "Sine"  # Sine, ReLU and Tanh are available
model = CallOption(Xi, T, M, N, D, Mm, layers, mode, activation)

n_iter = 2 * 10 ** 4
lr = 1e-3
#%%
tot = time.time()
print(model.device)
graph = model.train(n_iter, lr)
print("total time:", time.time() - tot, "s")
#%%
# model.load_model("models/CallOption4-256XVAPaper.pth")
#%%
n_iter = 51 * 10 ** 2
lr = 1e-5
#%%
tot = time.time()
print(model.device)
graph = model.train(n_iter, lr)
print("total time:", time.time() - tot, "s")
#%%

np.random.seed(37)
t_test, W_test = model.fetch_minibatch()
X_pred, Y_pred = model.predict(Xi, t_test, W_test)

if type(t_test).__module__ != 'numpy':
    t_test = t_test.cpu().numpy()
if type(X_pred).__module__ != 'numpy':
    X_pred = X_pred.cpu().detach().numpy()
if type(Y_pred).__module__ != 'numpy':
    Y_pred = Y_pred.cpu().detach().numpy()

for i in range(15):
    t_test_i, W_test_i = model.fetch_minibatch()
    X_pred_i, Y_pred_i = model.predict(Xi, t_test_i, W_test_i)
    if type(X_pred_i).__module__ != 'numpy':
        X_pred_i = X_pred_i.cpu().detach().numpy()
    if type(Y_pred_i).__module__ != 'numpy':
        Y_pred_i = Y_pred_i.cpu().detach().numpy()
    if type(t_test_i).__module__ != 'numpy':
        t_test_i = t_test_i.cpu().numpy()
    t_test = np.concatenate((t_test, t_test_i), axis=0)
    X_pred = np.concatenate((X_pred, X_pred_i), axis=0)
    Y_pred = np.concatenate((Y_pred, Y_pred_i), axis=0)
X_pred = X_pred[:500, :]
# %%
from scipy.stats import multivariate_normal as normal

# %%
X_preds = X_pred[:, :, 0]


# %%
def black_scholes_call(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * np.exp(-q * T) * normal.cdf(d1)) - (K * np.exp(-r * T) * normal.cdf(d2))
    delta = normal.cdf(d1)
    return call_price, delta


def calculate_option_prices(X_pred, time_array, K, r, sigma, T, q=0):
    rows, cols = X_pred.shape
    option_prices = np.zeros((rows, cols))
    deltas = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            S = X_pred[i, j]
            t = time_array[j]
            time_to_maturity = T - t
            if time_to_maturity > 0:
                option_prices[i, j], deltas[i, j] = black_scholes_call(S, K, time_to_maturity, r, sigma, q)
            else:
                option_prices[i, j] = max(S - K, 0)
                if S > K:
                    deltas[i, j] = 1
                elif S == K:
                    deltas[i, j] = 0.5
                else:
                    deltas[i, j] = 0

    return option_prices, deltas


# Given parameters
K = 1.0  # Strike price
r = 0.01  # Risk-free interest rate
sigma = 0.25  # Volatility
q = 0  # Dividend yield (assuming none)
T = 1  # Expiry time in years

Y_test, Z_test = calculate_option_prices(X_preds, t_test[0], K, r, sigma, T, q)

errors = (Y_test[:500] - Y_pred[:500,:,0])**2
errors.mean(), errors.std()

np.sqrt(errors.mean())

graph = model.iteration, model.training_loss
#%%
def figsize(scale, nplots = 1):
    fig_width_pt = 438.17227
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale
    fig_height = nplots*fig_width*golden_mean
    fig_size = [fig_width,fig_height]
    return fig_size
#%%
plt.figure(figsize=figsize(1.0))
plt.plot(graph[0], graph[1])
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.yscale("log")
plt.title('Evolution of the training loss')
samples = 5
# plt.savefig('Figures/CallOption1DLoss.pdf')
plt.figure(figsize=figsize(1.0))
plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T)

plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T)

plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
plt.title(str(D) + '-dimensional Call Option, ' + model.mode + "-" + model.activation)

plt.show()

plt.figure(figsize=figsize(1.0))
plt.plot(t_test[0] * 100, Y_pred[0] * 100, 'b', label='Learned $u(t,X_t)$')
plt.plot(t_test[0] * 100, Y_test[0] * 100, 'r--', label='Exact $u(t,X_t)$')
plt.plot(t_test[0, -1] * 100, Y_test[0, -1] * 100, 'ko', label='$Y_T = u(T,X_T)$')
for i in range(7):
    plt.plot(t_test[i] * 100, Y_pred[i] * 100, 'b')
    plt.plot(t_test[i] * 100, Y_test[i] * 100, 'r--')
    plt.plot(t_test[i, -1] * 100, Y_test[i, -1] * 100, 'ko')
plt.plot([0], Y_test[0,0] * 100, 'ks', label='$Y_0 = u(0,X_0)$')
plt.title(str(D) + '-dimensional Call Option, ' + model.mode + "-" + model.activation)
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
plt.savefig("CallOption1DPreds.png")
# plt.show()
