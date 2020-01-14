# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:01:21 2020

@author: Jakob
"""

###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import time
import pickle

t1 = time.time()
fig = plt.figure()


# Print options
np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

# Gloabal parameter inputs
n = 50 # Number of agents
delta = 0.3 # weight placed on indirect links
gamma = 0.5 # weight placed on additional utility derived from a mutual link
c = 0.2 # cost of forming and maintaining links
b = 0.3 # strength of preference for links to similar agents 
sigma = 0.01 # standard deviation of the shocks to utility
alpha = 2.1 # convexity of costs

share_red = 1/3
share_blue = 1/3
share_green = 1-share_red-share_blue
possible_X = [[1, 0],[0, 1],[1,1]]

# Randomly generate the matrix of characteristics (Generating this randomly is NOT a good idea)
# Note that this way of generating guarantees that X_i=[0,0] does not occur
#X_ind = np.random.choice(len(possible_X), size=n, p=[share_red,share_blue,share_green])
#X = np.array([possible_X[X_ind[i]] for i in range(len(X_ind))])

# Generate proportional green blue and reds for sure (makes simulation more stable)
share_red = np.round(share_red, decimals=1)
share_blue = np.round(share_blue, decimals=1)
X = np.array([possible_X[0] for i in range(int(share_red*n))] + 
              [possible_X[1] for i in range(int(share_blue*n))] +
              [possible_X[2] for i in range(n-int(share_red*n)-int(share_blue*n))])

# Randomly generate the initial network configuration
p_link_0 = 0.1 # Uniform initial link probability
g_0 = np.random.choice([0, 1], size=(n,n), p=[1-p_link_0,p_link_0])
np.fill_diagonal(g_0, 0) # The diagonal elements of the adjacency matrix are 0 by convention
 # Sequence of adjacency matrices

def u(i,j,X):
    """ Returns the partial utility given X_i and X_j using the exp(-b*L1-norm
    of their difference)"""
    return math.exp(-b*np.linalg.norm((X[i] - X[j]), ord=1))

def make_pre_cal_u(n, X):
    """ Make the U matrix for the entire system
    """
    pre_cal_u = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            pre_cal_u[i, j] = u(i,j,X)

    return pre_cal_u

def U(g, i) :
    """ Returns the full utility of agent i given the current network structure
    g and the matrix of characteristics X """

    d_i = g[i].sum()
    direct_u = np.sum(g[i] * pre_U[i])
    mutual_u = np.sum(g[i] * g.T[i] * pre_U[i])
    indirect_u = np.sum((g.T.dot(g[i, :]) * pre_U)[i])

    return direct_u + gamma * mutual_u + delta * indirect_u - d_i ** alpha * c

def step(g,indexes):
    """ Randomly selects an agent i to revise their link with another random 
    agent j. Returns the updated adjacency matrix """

    eps = np.random.normal(scale=sigma, size=n*2)

    for i in indexes:
        r1 = i
        while r1 == i:
            r1 = np.random.choice(indexes)

        g[i, r1] = 0
        U_without = U(g, i) + eps[i]
        g[i, r1] = 1
        U_with = U(g, i) + eps[-i]

        if U_without > U_with:
            g[i, r1] = 0
        else:
            g[i, r1] = 1

    return g




def plot_network(g):
    """ Uses networkX to plot the directed network g """
    rows, cols = np.where(g == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph() # Calling the DIRECTED graph method
    gr.add_nodes_from(range(n))
    gr.add_edges_from(edges)
    # Add node colors according to X
    color_map = []
    for i in range(n):
        if np.all(X[i]==possible_X[0]):
            color_map.append('red')
        if np.all(X[i]==possible_X[1]):
            color_map.append('blue')
        if np.all(X[i]==possible_X[2]):
            color_map.append('green')

    fig.clear()
    nx.draw(gr, node_color=color_map, with_labels=True, node_size=500)
    plt.pause(0.5)

# Run the simulation for T total steps or until convergence is reached
T = 10000
T = int(T/n)
t_plot = 49

pre_U = make_pre_cal_u(n, X)
indexes = list(range(n))

g_sequence = np.zeros((T, n,n))
zero_sequence = np.zeros(T)
g_sequence[0] = g_0
g_old = g_0

# zeros = np.zeros(3)
for t in range(1, T):
    # print(t, end='\r')
    np.random.shuffle(indexes)

    # Perform a step and attach the new network
    g_new = step(g_old, indexes)
    g_sequence[t] = g_new
    zero_sequence[t] = (np.sum(g_sequence[t] - g_sequence[t-1]))

    print(zero_sequence[t])
    # try:
    #     if sum(zero_sequence[t-3:t]) == zeros:
    #         print(t, 'STOP')
    #         break
    # except:
    #     pass
    g_old = g_new

    # Produce a plot and diagnostics every t_plot steps
    if (t+1)%t_plot == 0:
        plot_network(g_new)




pickle.dump(g_sequence, open('gsec.p', 'wb'))
print(time.time() - t1)
plot_network(g_sequence[-2])
print("done")
plt.show()
