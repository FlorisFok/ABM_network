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
np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

# Gloabal parameter inputs
n = 10 # Number of agents
delta = 0.3 # weight placed on indirect links
gamma = 0.5 # weight placed on additional utility derived from a mutual link
c = 0.2 # cost of forming and maintaining links
b = 0.3 # strength of preference for links to similar agents 
sigma = 0 # standard deviation of the shocks to utility
tau = 2.1 # convexity of costs

share_red = 1/3
share_blue = 1/3
share_green = 1-share_red-share_blue

# Randomly generate the matrix of characteristics (Generating this randomly is NOT a good idea)
# Note that this way of generating guarantees that X_i=[0,0] does not occur
#possible_X = [[1, 0],[0, 1],[1,1]]
#X_ind = np.random.choice(len(possible_X), size=n, p=[share_red,share_blue,share_green])
#X = np.array([possible_X[X_ind[i]] for i in range(len(X_ind))])

# Generate proportional green blue and reds for sure (makes simulation more stable)
share_red = np.round(share_red, decimals=1)
share_blue = np.round(share_blue, decimals=1)
X = np.array([possible_X[0] for i in range(int(share_red*n))] + 
              [possible_X[1] for i in range(int(share_blue*n))] +
              [possible_X[2] for i in range(n-int(share_red*n)-int(share_blue*n))])

# Randomly generate the initial network configuration
p_link_0 = 0 # Uniform initial link probability
g_0 = np.random.choice([0, 1], size=(n,n), p=[1-p_link_0,p_link_0])
np.fill_diagonal(g_0, 0) # The diagonal elements of the adjacency matrix are 0 by convention
g_sequence = [g_0] # Sequence of adjacency matrices



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
            
    return direct_u + gamma*mutual_u + delta*indirect_u - d_i**tau*c

def step(g,X):
    """ Randomly selects an agent i to revise their link with another random 
    agent j. Returns the updated adjacency matrix """
    i = np.random.choice(range(n))
    j = np.random.choice([x for x in range(n) if x!=i]) # select from agents other than i
    g_ij_initial = g[i,j]

    eps = np.random.normal(scale=sigma,size=2) # Simulate two shocks from normal with std dev sigma
    
    g[i,j] = 1
    U_with_link = U(i,g,X) + eps[0]
    g[i,j] = 0
    U_without_link = U(i,g,X) + eps[1]
    
    if U_with_link > U_without_link:
        g[i,j] = 1
#        if g_ij_initial==1:
#            print('Link kept')
#        if g_ij_initial==0:
#                print('Link formed')
    if U_with_link == U_without_link:
        g[i,j] = g_ij_initial
#        print('Status quo')
    if U_with_link < U_without_link:
        g[i,j] = 0
#        if g_ij_initial==1:
#            print('Link destroyed')
#        if g_ij_initial==0:
#            print('No link formed')
#    print(g)
    return g
    


def plot_network(g):
    """ Uses networkX to plot the directed network g """
    rows, cols = np.where(g == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph() # Calling the DIRECTED graph method
    gr.add_edges_from(edges)
    # Add node colors according to X
    color_map = []
    for i in gr.nodes:
        if np.all(X[i]==[1,0]):
            color_map.append('red')
        if np.all(X[i]==[0,1]):
            color_map.append('blue')
        if np.all(X[i]==[1,1]):
            color_map.append('green')
    nx.draw(gr, node_color=color_map, with_labels=True, node_size=500)
    
    plt.show()


# Run the simulation for T total steps
T=1000
t_plot = 100 # Produce a plot and diagnostics every t_plot steps
for t in range(T-1):
    g_new = step(g_sequence[-1],X)
    g_sequence.append(g_new)
    if t%t_plot == 0 and t/t_plot>0:
        plot_network(g_sequence[-1])
#        print(np.linalg.norm((g_sequence[t+1]-g_sequence[t+1-t_plot]), ord=1))
        degrees = [sum(g_sequence[-1][i]) for i in range(n)]
        print('Mean, variance of degree: {},{}'
              .format(np.mean(degrees),np.var(degrees)))
#        print('Average degree for reds blues, and greens: {},{},{}'
#              .format(np.mean([sum(g_sequence[-1][i]) for i in range(n)])))
    
