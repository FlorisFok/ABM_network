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
missing_list = []

# Read in node features
for school in range(1,n_schools+1):
    # skip school 1 and 48 as it does not exist/has weird data
    if school == 1 or school == 48:
        continue
    
    url = 'http://moreno.ss.uci.edu/comm' + str(school)
    url_X = url + '_att.dat'

    print("Converting data of school {}".format(school))

    X = pd.read_csv(url_X, delim_whitespace=True, names=['sex','race','grade','school'], skiprows=8)
    # remove top and bottom lines if they don't contain numbers < the number of rows before the data is not always equal
    for i in range(5):
        if np.isnan(X.grade.iloc[0]):
            X = X.iloc[1:]
        if np.isnan(X.grade.iloc[-1]):
            X = X.iloc[:-1]
        
    n = len(X)
    
    # change the value of the school_ID to 0 if it were nan
    X.school.fillna(0, inplace=True)
    
    # the last row of school 35 contains a "?", which should be converted into 0
    if school == 35:
        X.race.iloc[-1] = 0
        
    X = X.astype('int8')
    
    
    # Read in network structure
    url_g = url + '.dat'
    df_g = pd.read_csv(url_g, delim_whitespace=True,
                       names=['ego','alter','tie_strength'], skiprows=4)


    g = np.zeros((n,n))
    for row in range(len(df_g)):
        i = df_g.ego.iloc[row]-1 # -1 because the data does not have indeces starting from 0
        j = df_g.alter.iloc[row]-1
        try:
            g[i,j] = 1
        except Exception as e:
            print(e) # for infrequently encountered out of bound errors, i.e.
            # invalid nomination ids
            
            
    # Check average degree to make sure everything is working as intended
    print('Avg degree is {}'.format(sum([sum(g[i]) for i in range(n)]/n)))

    # Add 1 to school
    X.school = X.school + 1
    
    full_ind = X.index
    
    # Remove rows of X with missing values (0)
    X = X.loc[(X!=0).all(1)]
    
    complete_ind = X.index
    
    missing_ind = [i for i in full_ind if i not in complete_ind]
    
    for i in missing_ind:
        try:
            g = np.delete(g, i, axis=0)
            g = np.delete(g, i, axis=1)
        except Exception as e:
            print(e) # out of bound errors
    

    # If length of X and g are not equal, remove last obs from the longer dataset
    g_X_diff = len(g)-len(X)
    print(g_X_diff)
    if g_X_diff > 0:
        g = g[:-abs(g_X_diff),:-abs(g_X_diff)]
    if g_X_diff < 0:
        X = X[:-abs(g_X_diff)]
        
    g = g.astype('int8')
    
    print('Community {}: {} observations and  {} missing'.format(school,
          len(X), n-len(X)))
  
    print(X.min())
    X_list.append(X)
    g_list.append(g)
    missing_list.append(n-len(X))
    
print('{} is the average share of missing observations.'.format(
        np.average([missing_list[i]/len(X_list[i]) for i in range(len(missing_list))])))


#np.unique(np.where(X[['sex','race','grade']] == 0))
#for school in range(len(X_list)):
#    print(sum([X_list[school].iloc[i]['sex','race','grade']==0 for i in range(len(X_list[school]))]))
#    print(range(len(X_list[school])),'\n')
#    

# Save list of X matrices as a pickle
with open('X_list.pkl', 'wb') as f:
    pickle.dump(X_list, f)
   

# Save list of g matrices as a pickle
with open('g_list.pkl', 'wb') as f:
    pickle.dump(g_list, f)  
    
# Save list of missing as a pickle
with open('missing_list.pkl', 'wb') as f:
    pickle.dump(missing_list, f) 
    
    

    
# Descriptive statistics
#for i in range(len(X_list)):
#    print('\nSchool {}'.format(i))
#    print(X_list[i].describe())

#small_sample_ind = [67,68,74]
#X_small_sample = [X_list[i] for i in small_sample_ind]
#g_small_sample = [g_list[i] for i in small_sample_ind]
#
#with open('g_sample_list.pkl', 'wb') as f:
#    pickle.dump(g_small_sample, f)
#
#with open('X_sample_list.pkl', 'wb') as f:
#    pickle.dump(X_small_sample, f)    
#    
## To read later, use:
#with open('g_list.pkl', 'rb') as f:
#    g_list = pickle.load(f)
#
#
## Plot the generated networks
#def plot_network(g, color_map):
#    """ Uses networkX to plot the directed network g """
#    n = len(g)
#    rows, cols = np.where(g == 1)
#    edges = zip(rows.tolist(), cols.tolist())
#    gr = nx.DiGraph() # Calling the DIRECTED graph method
#    gr.add_nodes_from(range(n))
#    gr.add_edges_from(edges)
#
#    nx.draw(gr, node_color=color_map, with_labels=False, node_size=50)
#
#    plt.show()
#
#
#for comm in range(len(small_sample_ind)):
#    # Add node colors according to X
#    color_map = []
#    for i in range(len(g_small_sample[comm])):
#        if X_small_sample[comm].sex.iloc[i] == 1:
#            color_map.append('red')
#        if X_small_sample[comm].sex.iloc[i] == 2:
#            color_map.append('blue')
#
#    plot_network(g_small_sample[comm], color_map)


