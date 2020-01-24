import numpy as np

def network_statistics(g):
    ''' Returns the following summary statistics for a directed network adjacency
    matrix g: density at maximum reach (RCHDEN), relative density (RELDEN),
    proportion symmetric dyads (PTCMUT), and mutuality index (RHO2) '''

    n = len(g)    
    
    #density DENX2
    #DENX2 = np.sum(g)/(n*(n-1))
    
    
    # density at maximum reach RCHDEN
    
    # define the function to tranfer adjacency matrix to reachability matrix  
    # Prints reachability matrix of graph[][] using Floyd Warshall algorithm 
    # function found on https://www.geeksforgeeks.org/transitive-closure-of-a-graph/
    '''reach[][] will be the output matrix that will finally 
    have reachability values. 
    Initialize the solution matrix same as input graph matrix'''
    reach =[i[:] for i in g] 
    '''Add all vertices one by one to the set of intermediate 
    vertices. 
    ---> Before start of a iteration, we have reachability value 
    for all pairs of vertices such that the reachability values 
    consider only the vertices in set  
    {0, 1, 2, .. k-1} as intermediate vertices. 
    ----> After the end of an iteration, vertex no. k is 
    added to the set of intermediate vertices and the  
    set becomes {0, 1, 2, .. k}'''
    for k in range(n): 
              
        # Pick all vertices as source one by one 
        for i in range(n): 
                  
            # Pick all vertices as destination for the 
            # above picked source 
            for j in range(n): 
                      
                # If vertex k is on a path from i to j,  
                    # then make sure that the value of reach[i][j] is 1 
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j]) 
    
    RCHDEN = np.sum(reach)/(n*(n-1))
    
    
    # relative density RELDEN
    RELDEN = np.sum(g)/(10*n) # 10*n is the maximum nomber of ties given that
    # the maximum number of friendship nominations is 10


    # create upper triangular matrix with 2's on mutual dyads, 1's on asymmetric
    # dyads and count occurrence
    added_up = np.triu(g + np.transpose(g))
    mutual_d = np.count_nonzero(added_up == 2)
    asym_d = np.count_nonzero(added_up == 1)
    total_d = mutual_d + asym_d
    
    # calculate proportion symmetric dyads (PTCMUT) and asymmetric dyads (PTCASY)
    PTCMUT = mutual_d / total_d
    #PTCASY = asym_d / total_d
    
    
    # count total out_degree connections
    out_degree = g.sum()
    # take the sum of squares of the out degree connections per individual (row)
    sum_squares_out = (g.sum(axis=1)**2).sum()
    
    # calculate mutuality index (RHO2) (according to Katz and Powell’s (1955))
    RHO2 = (2*(n - 1)**2 * mutual_d - out_degree**2 + sum_squares_out)/(out_degree
           *(n - 1)**2 - out_degree**2 + sum_squares_out)
    
    # determine the local clustering coefficient mean and standard deviation
    clustering_coefficients = []
    for n_node, connections in enumerate(g):
        # the amount of neighbours each node has
        n_neighbours = np.sum(g[n_node])
        # only consider n with at least 2 neighbours
        if n_neighbours >= 2:
            # matrix of the n that are both neighbours of the node considered
            neighbour_matrix = np.dot(np.transpose([g[n_node]]),[g[n_node]])
            # the amount of connections between neighbours
            neighbour_connections = np.sum(g*neighbour_matrix)
            # the amount of connections between neighbours divided by the possible amount of connections
            clustering_coefficients.append(neighbour_connections
                                           /(n_neighbours*(n_neighbours-1)))
    mean_clustering_coefficient = np.mean(clustering_coefficients)
    std_clustering_coefficient = np.std(clustering_coefficients)
    clustering_coefficient = [mean_clustering_coefficient,std_clustering_coefficient]

    
    return RCHDEN, RELDEN, PTCMUT, RHO2, clustering_coefficient