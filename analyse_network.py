import numpy as np
import math

def determine_outdegree(connectivity):
    # this returns the outdegree 
    # and correlation of the outdegree of neighbours

    # total number of connections
    mean_connections = np.sum(connectivity)/len(connectivity)
    # mean corrected connectivity vector
    connectivity_vector = np.zeros(len(connectivity))
    mc_connectivity_vector = np.zeros(len(connectivity))
    for n_student, connections in enumerate(connectivity):
        connectivity_vector[n_student] = np.sum(connections)
        mc_connectivity_vector[n_student] = np.sum(connections)-mean_connections   
    # outdegree correlation matrix for all nodes (so even unconnected)
    outdeg_corr = np.dot(np.transpose([mc_connectivity_vector]),[mc_connectivity_vector])
    # outdegree correlation matrix for neighbours
    neigh_outdeg_corr = outdeg_corr * connectivity
    # mean and standard deviation of the outdegree correlation of neighbours
    mean_outdeg = np.mean(connectivity_vector)
    std_outdeg = np.std(connectivity_vector)
    mean_outdeg_corr = np.mean(neigh_outdeg_corr[np.nonzero(neigh_outdeg_corr)])
    std_outdeg_corr = np.std(neigh_outdeg_corr[np.nonzero(neigh_outdeg_corr)])
    
    return mean_outdeg,std_outdeg,mean_outdeg_corr,std_outdeg_corr

def analyse(connectivity, characteristics):
    """
        Calculate mean and standard deviation (output in tuples [mean,sd]) of following measures:
        
        - degree
        - mut_prop
        - cluster_coef
        - segreg_ind    > list with length of the amount of characteristics (3 for sex, race, grade)
        
        INPUT: 
        - connectivity matrix with row-students nominating column-students as friends
        - characteristics matrix with row per student, with integers indicating every group for each characteristic (sex, race, grade)
    """
    
    # get amount of nodes and list of out going dyads for every individual
    nodes = connectivity.shape[0]
    out_d = np.count_nonzero(connectivity, axis=1)
    
    
    # determine degree nodes (outgoing connections)
    mean_degree = np.mean(out_d)
    std_degree = np.std(out_d)
    degree = [mean_degree, std_degree]

    
    # determine the mutual dyads proportion
    # create matrix with 2's on mutual dyads, 1's on asymmetric dyads and count occurrence
    added_up = connectivity + np.transpose(connectivity)
    mutual_d = np.count_nonzero(added_up == 2, axis=1)
    mut_prop = mutual_d / out_d
    # remove 'nan' individuals (with no out-going connections) from list
    mut_prop = [value for value in mut_prop if not math.isnan(value)]
    # calculate mean+std mutual dyads proportion
    mean_mut_prop = np.mean(mut_prop)
    std_mut_prop = np.std(mut_prop)
    mut_prop = [mean_mut_prop, std_mut_prop]
    
    
    # determine the local clustering coefficient
    clustering_coefficients = []
    for n_node, connections in enumerate(connectivity):
        # the amount of neighbours each node has
        n_neighbours = np.sum(connectivity[n_node])
        # only consider nodes with at least 2 neighbours
        if n_neighbours >= 2:
            # matrix of the nodes that are both neighbours of the node considered
            neighbour_matrix = np.dot(np.transpose([connectivity[n_node]]),[connectivity[n_node]])
            # the amount of connections between neighbours
            neighbour_connections = np.sum(connectivity*neighbour_matrix)
            # the amount of connections between neighbours divided by the possible amount of connections
            clustering_coefficients.append(neighbour_connections / (n_neighbours*(n_neighbours-1)))
    # calculate mean+std clustering coefficient
    mean_cluster_coef = np.mean(clustering_coefficients)
    std_cluster_coef = np.std(clustering_coefficients)
    cluster_coef = [mean_cluster_coef, std_cluster_coef]

    
    # determine the segregation index per characteristic (sex, race, grade)
    segreg_ind = []
    # iterate through different characteristics (sex, race, grade)
    for i in range(characteristics.shape[1]):
        # get different groups of this characteristic in dataset
        characs = sorted(list(set(characteristics.ix[:,i])))
        amount = len(characs)
        # for every characteristic own tuple for mean and std
        segreg_ind_charac = []
        # iterate through different groups of this characteristic
        for j in range(amount):
            # indicate indices of members this group and save size group
            indices = np.where(characteristics.ix[:,i] == characs[j])[0]
            # calculate ratio out-group individuals
            ratio_diff = 1 - len(indices) / nodes
            # create a submatrix of all nominations from this group and save amount
            submat_trait = connectivity[np.ix_(indices,)]
            # create submatrix outgoing connections to individuals different group
            mask = np.ones(connectivity.shape[0], np.bool)
            mask[indices] = 0
            submat_diff = submat_trait[:,mask]
            # calculate segregation index per individual of this group for this characteristic
            for ind in range(len(indices)):
                expect_out = submat_trait[ind].sum() * ratio_diff
                observ_out = submat_diff[ind].sum()
                seg_ind = (expect_out - observ_out) / expect_out
                if seg_ind < -1:
                    seg_ind = -1
                segreg_ind_charac.append(seg_ind)
        # remove 'nan' individuals from list
        segreg_ind_charac = [value for value in segreg_ind_charac if not math.isnan(value)]
        # calculate mean+std segregation index this characteristic
        mean_segreg_ind_charac = np.mean(segreg_ind_charac)
        std_segreg_ind_charac = np.std(segreg_ind_charac)
        segreg_ind.append([mean_segreg_ind_charac, std_segreg_ind_charac])
    
    
    return degree, mut_prop, cluster_coef, segreg_ind[0], segreg_ind[1], segreg_ind[2]