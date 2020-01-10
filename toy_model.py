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
# Print options
np.set_printoptions(precision=4, threshold=10000, linewidth=150, suppress=True)

# Gloabal parameter inputs
n = 10 # Number of agents
delta = 0.5 # weight placed on indirect links
gamma = 0.2 # weight placed on additional utility derived from a mutual link
c = 0.5 # cost of forming and maintaining links
b = 1 # strength of preference for links to similar agents 
sigma = 0.2 # standard deviation of the shocks to utility

# Randomly generate the initial network configuration
p_link_0 = 0.5 # Uniform initial link probability
g_0 = np.random.choice([0, 1], size=(n,n), p=[1-p_link_0,p_link_0])
np.fill_diagonal(g_0, 0) # The diagonal elements of the adjacency matrix are 0 by convention

# Randomly generate the matrix of characteristics
share_red = 1/4
share_blue = 1/4
share_green = 1-share_red-share_blue
# Note that this way of generating guarantees that X_i=[0,0] does not occur
possible_X = [[0, 1],[1, 0],[1,1]]
X_ind = np.random.choice(len(possible_X), size=n, p=[share_red,share_blue,share_green])
X = np.array([possible_X[X_ind[i]] for i in range(len(X_ind))])


def u(i,j,X):
    """ Returns the partial utility given X_i and X_j using the exp(-b*L1-norm
    of their difference)"""
    return math.exp(-b*np.linalg.norm((X[i] - X[j]), ord=1))

def U(i,g,X):
    """ Returns the full utility of agent i given the current network structure
    g and the matrix of characteristics X """
    d_i = sum(g[i]) # degree of i
    
    direct_u = sum([g[i,j]*u(i,j,X) for j in range(n)])

    mutual_u = sum([g[i,j]*g[j,i]*u(i,j,X) for j in range(n)])
 
    indirect_u = 0
    for j in range(n):
        for k in range(n):
            if k==i or k==j:
                continue
            else:
                indirect_u += g[i,j]*g[j,k]*u(i,k,X)
            
    return direct_u + gamma*mutual_u + delta*indirect_u - d_i*c

def step(g,X):
    """ Randomly selects an agent i to revise their link with another random 
    agent j. Returns the updated adjacency matrix """
    i = np.random.choice(range(n))
    j = np.random.choice([x for x in range(n) if x!=i]) # select from agents other than i
    
    eps = np.random.normal(scale=sigma,size=2) # Simulate two shocks from normal with std dev sigma
    
    g_ij_initial = g[i,j]
    g[i,j] = 1
    U_with_link = U(i,g,X) + eps[0]
    g[i,j] = 0
    U_without_link = U(i,g,X) + eps[1]
    
    if U_with_link > U_without_link:
        g[i,j] = 1
    if U_with_link == U_without_link:
        g[i,j] = g_ij_initial
    if U_with_link < U_without_link:
        g[i,j] = 0
    
    return g


def plot_network(g):
    """ Uses networkX to plot the directed network g """
    rows, cols = np.where(g == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph() # Calling the DIRECTED graph method
    gr.add_edges_from(edges)
    nx.draw(gr, with_labels=True, node_size=500)
    plt.show()

plot_network(g_0)

for t in range(T):
g_1step(g_0,X)
