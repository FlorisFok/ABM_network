# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:46:55 2020

@author: Jakob
"""

###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle

# Print options
np.set_printoptions(precision=4, threshold=10000, linewidth=150, suppress=True)

n_schools = 84

g_list = []
X_list = []

for school in range(n_schools):
    url = 'http://moreno.ss.uci.edu/comm' + str(school+1)
    
    # Read in network structure
    url_g = url + '.dat'
    df_g = pd.read_csv(url_g, delim_whitespace=True, 
                       names=['ego','alter','tie_strength'], skiprows=4)
    n = df_g.ego.max()
    g = np.zeros((n,n))
    for row in range(len(df_g)):
        i = df_g.ego.iloc[row]-1
        j = df_g.alter.iloc[row]-1
        try:
            g[i,j] = 1
        except Exception as e:
            print(e) # for infrequently encountered out of bound errors, i.e. 
            # invalid nomination ids
    
    print(sum([sum(g[i]) for i in range(n)]/n)) # Check average degree to make 
    # sure everything is working as intended
    
    g_list.append(g)
    
# Save list of g matrices as a pickle
with open('g_list.pkl', 'wb') as f:
    pickle.dump(g_list, f)

# To read later, use:
#with open('g_list.pkl', 'rb') as f:
#    g_list = pickle.load(f)
    
# Read in node features (this only works for the first file so far)
for school in range(1,n_schools+1):
    url = 'http://moreno.ss.uci.edu/comm' + str(school)
    
    url_X = url + '_att.dat'
    df_X = pd.read_csv(url_X, delim_whitespace=True, names=['feature','value'], skiprows=8)
    df_X = df_X.iloc[:-1]
    df_X.reset_index(inplace=True)
    df_X = df_X.astype('int32')
    X = np.zeros((n,3))
    for row in range(len(df_X)):
        i = df_X['index'].iloc[row]-1
        j = df_X.feature.iloc[row]-1
        X[i,j] = df_X.value.iloc[row]
    
    X_list.append(X)
    

# Plot the generated networks
def plot_network(g):
    """ Uses networkX to plot the directed network g """
    rows, cols = np.where(g == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph() # Calling the DIRECTED graph method
    gr.add_nodes_from(range(n))
    gr.add_edges_from(edges)

    nx.draw(gr, with_labels=False, node_size=100)
    
    plt.show()

for g in g_list:
    if len(g) < 5000:
        plot_network(g)
        
plot_network(g_list[0])
