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

# Read in node features
for school in range(1,n_schools+1):
    # skip school 48 as it does not exist
    if school != 48:
        #n = 71 # comment out if g_list is made
        url = 'http://moreno.ss.uci.edu/comm' + str(school)
        url_X = url + '_att.dat'

        print("converting X data of school {}".format(school))

        # do a different thing for the first school
        if school == 1:
            df_X = pd.read_csv(url_X, delim_whitespace=True, names=['feature', 'value'], skiprows=8)
            df_X = df_X.iloc[:-1]
            df_X.reset_index(inplace=True)
            df_X = df_X.astype('int32')
            X = np.zeros((n,3)) # you sure n is defined correctly? < Koen
            for row in range(len(df_X)):
                i = df_X['index'].iloc[row]-1
                j = df_X.feature.iloc[row]-1
                X[i,j] = df_X.value.iloc[row]
        else:
            df_X = pd.read_csv(url_X, delim_whitespace=True, names=['sex','race','grade','school'], skiprows=8)
            X = np.array(df_X)
            # remove top lines if they don't contain numbers < the number of rows before the data is not always equal
            for i in range(3):
                if not np.isfinite(X[0,2]):
                    X = np.delete(X, obj=0, axis=0)

            # change the value of the school_ID to 0 if it were nan
            if not np.isfinite(X[0,3]):
                X[:,3] = 0
            # the last row of school 35 contains a "?", which should be converted into 0
            if school == 35:
                print('X[-1,1] = {}'.format(X[-1,1]))
                X[-1,1] = 0
            X = X.astype('int32')

        # convert the race data to one variable per race
        race_numbers = X[:,1]
        dummy_race = pd.get_dummies(race_numbers)
        X = np.concatenate((X, dummy_race), axis=1)

    else:
        X = np.zeros((1,1))

    X_list.append(X)

# Save list of X matrices as a pickle
with open('X_list.pkl', 'wb') as f:
    pickle.dump(X_list, f)

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
