
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import sys
import copy
import os
import pickle

colors = ['#FF0000', '#00FF00', '#0000FF',
          '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0',
          '#808080', '#800000',	'#808000', '#008000',
          '#800080', '#008080', '#000080']


## Cost function #######
def cost(d_i) :
    return d_i ** ALPHA * C


## conv_rule ###########
def conv_rule(g_sequence, t) :
    return np.linalg.norm((g_sequence[t - 1] - g_sequence[t]), ord=1)
    # return (np.sum(g_sequence[t] - g_sequence[t - 1]))


## STOP RULE ###########
def stop_rule(zero_sequence, t) :
    return zero_sequence[t - n_zeros :t].any() == zeros.any()


def analyse_network(connectivity, characteristics) :
    # OUTPUT:
    #
    # average degree of nodes network            > av_degree
    # density at maximum reach (RCHDEN)          > den_max_reach
    # relative density (RELDEN)                  > rel_den
    # proportion symmetric dyads (PTCMUT)        > p_symm_dyads
    # mutuality index (RHO2)                     > mutuality_index
    # amount of clustering (groups of friends)   > clustering_coefficient   [mean, std dev]
    # homophily index per trait in dict          > homoph_ind               {"sex_", "race_", "grade_"}
    # segregation index per trait in tuple       > segreg_ind               [sex, race, grade]

    # INPUT:
    # - connectivity matrix with row-students nominating column-students as friends
    # - characteristics matrix with row per student, with integers indicating every group for each characteristic (sex, race, grade)
    charr = np.zeros((characteristics.shape[0], 3))

    for i in range(3) :
        charr[:, i] = characteristics.iloc[:, i]

    characteristics = charr

    nodes = connectivity.shape[0]
    mutual_d = 0
    asym_d = 0
    trait = ["sex", "race", "grade"]

    # density DENX2
    # DENX2 = np.sum(connectivity)/(nodes*(nodes-1))

    # density at maximum reach RCHDEN

    # define the function to tranfer adjacency matrix to reachability matrix
    # Prints reachability matrix of graph[][] using Floyd Warshall algorithm
    # function found on https://www.geeksforgeeks.org/transitive-closure-of-a-graph/
    reachability = copy.deepcopy(connectivity)
    '''reach[][] will be the output matrix that will finally 
    have reachability values. 
    Initialize the solution matrix same as input graph matrix'''
    reach = [i[:] for i in reachability]
    '''Add all vertices one by one to the set of intermediate 
    vertices. 
    ---> Before start of a iteration, we have reachability value 
    for all pairs of vertices such that the reachability values 
    consider only the vertices in set  
    {0, 1, 2, .. k-1} as intermediate vertices. 
    ----> After the end of an iteration, vertex no. k is 
    added to the set of intermediate vertices and the  
    set becomes {0, 1, 2, .. k}'''
    for k in range(nodes) :

        # Pick all vertices as source one by one
        for i in range(nodes) :

            # Pick all vertices as destination for the
            # above picked source
            for j in range(nodes) :
                # If vertex k is on a path from i to j,
                # then make sure that the value of reach[i][j] is 1
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    RCHDEN = np.sum(reach) / (nodes * (nodes - 1))

    # relative density RELDEN
    RELDEN = np.sum(connectivity) / (10 * nodes)

    # create upper triangular matrix with 2's on mutual dyads, 1's on asymmetric dyads and count occurrence
    added_up = np.triu(connectivity + np.transpose(connectivity))
    mutual_d = np.count_nonzero(added_up == 2)
    asym_d = np.count_nonzero(added_up == 1)
    total_d = mutual_d + asym_d

    # calculate proportion symmetric dyads (PTCMUT) and asymmetric dyads (PTCASY)
    PTCMUT = mutual_d / total_d
    # PTCASY = asym_d / total_d

    # count total out_degree connections
    out_degree = connectivity.sum()
    # average out_degree
    av_degree = out_degree / nodes
    # take the sum of squares of the out degree connections per individual (row)
    sum_squares_out = (connectivity.sum(axis=1) ** 2).sum()

    # calculate mutuality index (RHO2) (according to Katz and Powell’s (1955))
    RHO2 = (2 * (nodes - 1) ** 2 * mutual_d - out_degree ** 2 + sum_squares_out) / (
                out_degree * (nodes - 1) ** 2 - out_degree ** 2 + sum_squares_out)

    # determine the local clustering coefficient mean and standard deviation
    clustering_coefficients = []
    for n_node, connections in enumerate(connectivity) :
        # the amount of neighbours each node has
        n_neighbours = np.sum(connectivity[n_node])
        # only consider nodes with at least 2 neighbours
        if n_neighbours >= 2 :
            # matrix of the nodes that are both neighbours of the node considered
            neighbour_matrix = np.dot(np.transpose([connectivity[n_node]]), [connectivity[n_node]])
            # the amount of connections between neighbours
            neighbour_connections = np.sum(connectivity * neighbour_matrix)
            # the amount of connections between neighbours divided by the possible amount of connections
            clustering_coefficients.append(neighbour_connections / (n_neighbours * (n_neighbours - 1)))
    mean_clustering_coefficient = np.mean(clustering_coefficients)
    std_clustering_coefficient = np.std(clustering_coefficients)
    clustering_coefficient = [mean_clustering_coefficient, std_clustering_coefficient]

    # create homophily index dictionary and lists for the segregation index
    homoph_ind = dict()
    segreg_ind = []

    # iterate through different characteristics (sex, race, grade)
    for i in range(characteristics.shape[1]) :
        # get different groups of this characteristic in dataset
        characs = sorted(list(set(characteristics[:, i])))
        amount = len(characs)

        # counters for individuals per characteristic group and out-group nominations
        charac_count = []
        charac_out = []

        # counters for expected and observed cross trait nominations
        exp_cross = 0
        obs_cross = 0

        # iterate through different groups of this characteristic
        for j in range(amount) :
            # indicate indices of members this group and save size group
            indices = np.where(characteristics[:, i] == characs[j])[0]
            charac_count.append(len(indices))

            # create a submatrix of all nominations from this group and save amount
            submat_trait = connectivity[np.ix_(indices, )]
            charac_out.append(submat_trait.sum())

            # create 2 submatrices outgoing connections: 1 to individuals same group and 1 to individuals different group
            submat_same = connectivity[np.ix_(indices, indices)]
            mask = np.ones(connectivity.shape[0], np.bool)
            mask[indices] = 0
            submat_diff = submat_trait[:, mask]

            # count amount outgoing connections to same and to different group
            out_same = np.mean(submat_same.sum(axis=1))
            out_diff = np.mean(submat_diff.sum(axis=1))

            # add amount of cross trait nominations of this group to total
            obs_cross += submat_diff.sum()

            # calculate and save homophily index from this group for this characteristic
            homoph_ind[str(trait[i]) + "_" + str(characs[j])] = out_same / (out_same + out_diff)

        # calculate expected cross trait nominations (added for all combinations of traits (except with self))
        for row in range(amount) :
            for col in range(amount) :
                if row == col :
                    pass
                else :
                    exp_cross += charac_out[row] * (
                                (charac_count[row] * charac_count[col]) / (charac_count[row] * (nodes - 1)))

        # calculate and save segregation index for this trait
        if exp_cross == 0 :
            segreg_ind.append(9999)
            if amount == 1 :
                warnings.warn("Just one group: no segregation index for trait " + str(trait[i]) + "possible.")
            else :
                warnings.warn("Expected cross trait nominations is zero: no segregation index for trait " + str(
                    trait[i]) + "possible.")
        else :
            segreg_ind.append((exp_cross - obs_cross) / exp_cross)

    # return av_degree, RCHDEN, RELDEN, RHO2, segreg_ind[0], segreg_ind[1], segreg_ind[2]
    if clustering_coefficient[0] != clustering_coefficient[0]:
        clustering_coefficient[0] = 0

    return av_degree, RHO2, clustering_coefficient[0], segreg_ind[0], segreg_ind[1], segreg_ind[2]


class ConnectionMatrix:

    def __init__(self, n, p_link_0):
        self.minimal = 1
        self.n = n
        self.link_prop = p_link_0

        self.g = self.make_g_0()
        self.age = np.zeros((n, n)) + 1
        self.age_update()

    def make_g_0(self):
        """
        Initialize the first connections
        """
        g_0 = np.random.choice([0, 1], size=(self.n, self.n),
                               p=[1 - self.link_prop,
                               self.link_prop])

        np.fill_diagonal(g_0, 0)
        return g_0

    def age_update(self):
        self.age *= self.g
        self.age += self.g


class Model:

    def __init__(self, g, n, X, pos_X):
        self.n = n
        self.pos_X = pos_X
        self.g = g

        self.X = X
        self.U = self.make_pre_cal_U()
        self.P = self.make_prop()

        self.indexes = list(range(n))
        self.g_sequence = None
        self.zero_sequence = None

    def make_prop(self):
        # Setup probability matrix
        prop = np.zeros((self.n, self.n))

        # Loop over the person and their peers
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    prop[i, j] = 0
                else:
                    prop[i, j] = self.U[i, j] + MIN_PROP

            # Normalize
            prop[i, :] = prop[i, :] / np.sum(prop[i, :])

        return prop

    def make_pre_cal_U(self):
        """ Make the U matrix for the entire system
        """
        # Setup U
        pre_cal_u = np.zeros((self.n, self.n))

        race = list(self.X['race'])
        sex = list(self.X['sex'])
        grade = list(self.X['grade'])

        # Fill U
        for i in range(self.n):
            for j in range(self.n):
                pre_cal_u[i, j] = math.exp(- B1 * abs(sex[i] - sex[j])
                                           - B2 * abs(grade[i] - grade[j])
                                           - B3 * (0 if race[i] == race[j] else 1))

        return pre_cal_u

    def U_of_matrix(self, i):
        """ Returns the full utility of agent i given the current network structure
        g and the matrix of characteristics X """

        # degree, connection gain and cost calculations
        d_i = self.g.g[i].sum()
        direct_u = np.sum(self.g.g[i] * self.U[i])
        mutual_u = np.sum(self.g.g[i] * self.g.g.T[i] * self.U[i])

        # indirect connection gain
        a = (self.g.g.T.dot(self.g.g[i, :]) * self.U)[i]
        a[i] = 0
        indirect_u = np.sum(a)

        return direct_u + GAMMA * mutual_u + DELTA * indirect_u - cost(d_i)

    def step(self):
        """ Randomly selects an agent i to revise their link with another random
        agent j. Returns the updated adjacency matrix """

        # Add noise and shuffle indexes
        eps = np.random.normal(scale=SIGMA, size=self.n*2)
        np.random.shuffle(self.indexes)

        for i in self.indexes:
            # Choose new connection
            r1 = np.random.choice(self.indexes, p=self.P[i])

            # find value for new connection and removed connection
            self.g.g[i, r1] = 0
            U_without = self.U_of_matrix(i) + eps[i]

            self.g.g[i, r1] = 1
            U_with = self.U_of_matrix(i) + eps[-i]

            # Evaluate better option
            if U_without > U_with:
                self.g.g[i, r1] = 0
            else:
                self.g.g[i, r1] = 1

    def plot_network(self, final=False):
        """ Uses networkX to plot the directed network g """
        rows, cols = np.where(self.g.g == 1)  # returns row and column numbers where an edge exists

        # MAke the network
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_nodes_from(range(self.n))
        gr.add_edges_from(edges)
        # fig.clear()

        race = list(self.X['race'])
        sex = list(self.X['sex'])
        grade = list(self.X['grade'])

        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Add node colors according to X
        ax1.set_title('sex')
        color_map = []
        for i in range(self.n):
            for j, unit in enumerate(set(sex)):
                if np.all(sex[i] == unit):
                    color_map.append(colors[j])
        nx.draw(gr, ax=ax1, node_color=color_map, with_labels=False, node_size=100)

        ax2.set_title('race')
        color_map = []
        for i in range(self.n) :
            for j, unit in enumerate(set(race)) :
                if np.all(race[i] == unit) :
                    color_map.append(colors[j])
        nx.draw(gr, ax=ax2, node_color=color_map, with_labels=False, node_size=100)

        ax3.set_title('grade')
        color_map = []
        for i in range(self.n) :
            for j, unit in enumerate(set(grade)) :
                if np.all(grade[i] == unit) :
                    color_map.append(colors[j])
        nx.draw(gr, ax=ax3, node_color=color_map, with_labels=False, node_size=100)

        if not final:
            plt.pause(2)
        else:
            plt.pause(100)

    def save2pickle(self, pickle_name):
        """
        Saves data from the simulation to a pickle file.
        :param pickle_name: Name of pickle file
        """
        individuals_friendships_utilities = [self.X, self.g.g, self.U]
        pickle.dump(individuals_friendships_utilities, open(pickle_name, "wb"))

    def run(self, total_time, t_plot=0):
        """
        Run the ABM simulation
        :param total_time: Number of environment evaluations
        :param t_plot: Number of times the intermediate results need to be plotted
        :return: None (use plot to show results or save to save them)
        """
        if t_plot == 0:
            t_plot = total_time

        self.g_sequence = np.zeros((total_time, self.n, self.n))
        self.g_sequence[0] = self.g.g

        self.zero_sequence = np.zeros(total_time)
        self.zero_sequence[0] = 1.0

        for t in range(1, total_time):
            # Perform a step and attach the new network
            self.step()
            # print('step:', t, end='\r')

            self.g_sequence[t] = self.g.g
            self.zero_sequence[t] = conv_rule(self.g_sequence, t)

            if t > MINIMAL and stop_rule(self.zero_sequence, t):
                break

            # Produce a plot and diagnostics every t_plot steps
            if t % t_plot == 0:
                print("degree:", np.sum(self.g.g))
                self.plot_network()

        return t

    def rank(self):
        """
        Rank the connections you have
        """

        value_of_con = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                old = self.g.g[i][j]

                self.g.g[i][j] = 0
                not_conn = self.U_of_matrix(i)

                self.g.g[i][j] = 1
                conn = self.U_of_matrix(i)

                self.g.g[i][j] = old
                value_of_con[i][j] = not_conn - conn

        sex = list(self.X['sex'])

        def sort_value(x):
            return x[1]

        for i in range(self.n):
            male = []
            female = []
            for j in range(self.n):
                if self.g.g[i][j] != 0:
                    try:
                        if sex[i] == 1:
                            male.append((j, value_of_con[i][j]))
                        else:
                            female.append((j, value_of_con[i][j]))
                    except:
                        pass
            try:
                female.sort(key=sort_value, reverse=True)
                male.sort(key=sort_value, reverse=True)

                remain = female[:MAX_FRIEND]+male[:MAX_FRIEND]
                ind, score = zip(*remain)
            except:
                continue

            for friend in range(self.n):
                try:
                    if not friend in ind:
                        self.g.g[i][friend] = 0
                except:
                    pass




def read_excel_settings(loc):
    df = pd.read_excel(loc)
    settings_dict = {}

    for col in df:
        column = df[col]
        column = [i for i in column if i == i]
        if len(column) == 1:
            try:
                settings_dict[col] = float(column[0])
            except:
                settings_dict[col] = column[0]

        elif len(column) > 1:
            mat = [[]]
            for i in column:
                if i == '//':
                    mat.append([])
                else:
                    mat[-1].append(float(i))

            if len(mat) == 1:
                mat = mat[0]

            settings_dict[col] = mat

    return settings_dict

def avg(l, reruns):
    return [sum(l[i*reruns:(i+1)*reruns])/reruns for i in range(int(len(l)/reruns))]

def main(settings, MAX_LEN):
    global DELTA, GAMMA, C, B1, B2, B3, SIGMA, ALPHA, MIN_PROP, pos_link, MINIMAL, MAX_FRIEND, zeros, n_zeros
    DELTA, GAMMA, C, SIGMA, B1, B2, B3 = settings

    # Plot info
    plot_data = {'av_degree':[], 'clustering_coefficient':[], 'RHO2':[],
                 'segreg_ind0':[], 'segreg_ind1':[], 'segreg_ind2':[]}

    _string = ['av_degree', 'RHO2', 'clustering_coefficient', 'segreg_ind0', 'segreg_ind1', 'segreg_ind2']

    # True
    # g_matrix = pickle.load(open(r"C:\Users\FlorisFok\Downloads\g_list.pkl", 'rb'))
    big_x = pickle.load(open(r"C:\Users\FlorisFok\Downloads\x_list.pkl", 'rb'))

    MAX_FRIEND = 5
    MIN_PROP = 9
    pos_link = 0.1
    runs = 1000
    ALPHA = 2

    n_zeros = 3
    zeros = np.zeros(n_zeros)
    MINIMAL = n_zeros
    ########################

    models = []
    for model_num in range(len(big_x)):
        X = big_x[model_num]
        possible_X = [i[0] for i in list(X.groupby(['sex', 'race']))]
        n_agents = len(X['sex'])

        if n_agents > MAX_LEN:
            continue

        # Make and run model
        g = ConnectionMatrix(n_agents, pos_link)
        M = Model(g, n_agents, X, possible_X)
        M.run(runs, 0)
        M.rank()

        output = analyse_network(M.g.g, X)
        models.append(model_num)
        for s, o in zip(_string, output):
            plot_data[s].append(o)

    return plot_data

if __name__ == "__main__":
    from scipy.optimize import minimize
    import pickle

    settings = [0.2, 0.7, 0.125, 0.05, 0.3, 0.1, 0.01, 2.1]
    #  DELTA, GAMMA, C, B1, B2, B3, SIGMA, ALPHA
    # Constraints
    cons = ({'type' : 'ineq', 'fun' : lambda setting: 5 - setting[0]},  # delta[0, 5]
            {'type' : 'ineq', 'fun' : lambda setting : setting[0] - 10 ** (-6)},
            {'type' : 'ineq', 'fun' : lambda setting : 1 - setting[1]},  # gamma[0, 1]
            {'type' : 'ineq', 'fun' : lambda setting : setting[1] - 10 ** (-6)},
            {'type' : 'ineq', 'fun' : lambda setting : 5 - setting[7]},  # alpha[1, 5]
            {'type' : 'ineq', 'fun' : lambda setting : setting[7] - 1 - 10 ** (-6)},
            {'type' : 'ineq', 'fun' : lambda setting : 5 - setting[2]},  # C[0, 5]
            {'type' : 'ineq', 'fun' : lambda setting : setting[2] - 1 - 10 ** (-6)},
            {'type' : 'ineq', 'fun' : lambda setting : 5 - setting[3]},  # b123[1, 5]
            {'type' : 'ineq', 'fun' : lambda setting : setting[3] - 1 - 10 ** (-6)},
            {'type' : 'ineq', 'fun' : lambda setting : 5 - setting[4]},  # b123[1, 5]
            {'type' : 'ineq', 'fun' : lambda setting : setting[4] - 1 - 10 ** (-6)},
            {'type' : 'ineq', 'fun' : lambda setting : 5 - setting[5]},  # b123[1, 5]
            {'type' : 'ineq', 'fun' : lambda setting : setting[5] - 1 - 10 ** (-6)},
            {'type' : 'ineq', 'fun' : lambda setting : 5 - setting[6]},  # b123[1, 5]
            {'type' : 'ineq', 'fun' : lambda setting : setting[6] - 1 - 10 ** (-6)}
            )

    res = minimize(main, settings, method='SLSQP', constraints=cons)

    pickle.dump(res, open('res.p', 'wb'))

    print(res)  # check if it converged
    print(res.x)  # print results