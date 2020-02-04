
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import sys
import copy
import os
import pickle

## conv_rule ###########
def conv_rule(g_sequence, t):
    return np.linalg.norm((g_sequence[t - 1] - g_sequence[t]), ord=1)

## STOP RULE ###########
def stop_rule(zero_sequence, t) :
    return zero_sequence[t - n_zeros_conv :t].any() == zeros.any()


def analyse_network(connectivity, characteristics) :
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

    charr = np.zeros((characteristics.shape[0], 3))

    for i in range(3) :
        charr[:, i] = characteristics.iloc[:, i]

    characteristics = charr

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
    # calculate mean+std clustering coefficient
    mean_cluster_coef = np.mean(clustering_coefficients)
    std_cluster_coef = np.std(clustering_coefficients)
    cluster_coef = [mean_cluster_coef, std_cluster_coef]

    # determine the segregation index per characteristic (sex, race, grade)
    segreg_ind = []
    # iterate through different characteristics (sex, race, grade)
    for i in range(characteristics.shape[1]) :
        # get different groups of this characteristic in dataset
        characs = sorted(list(set(characteristics[:, i])))
        amount = len(characs)
        # for every characteristic own tuple for mean and std
        segreg_ind_charac = []
        # iterate through different groups of this characteristic
        for j in range(amount) :
            # indicate indices of members this group and save size group
            indices = np.where(characteristics[:, i] == characs[j])[0]
            # calculate ratio out-group individuals
            ratio_diff = 1 - len(indices) / nodes
            # create a submatrix of all nominations from this group and save amount
            submat_trait = connectivity[np.ix_(indices, )]
            # create submatrix outgoing connections to individuals different group
            mask = np.ones(connectivity.shape[0], np.bool)
            mask[indices] = 0
            submat_diff = submat_trait[:, mask]
            # calculate segregation index per individual of this group for this characteristic
            for ind in range(len(indices)) :
                expect_out = submat_trait[ind].sum() * ratio_diff
                observ_out = submat_diff[ind].sum()
                seg_ind = (expect_out - observ_out) / expect_out
                if seg_ind < -1 :
                    seg_ind = -1
                segreg_ind_charac.append(seg_ind)
        # remove 'nan' individuals from list
        segreg_ind_charac = [value for value in segreg_ind_charac if not math.isnan(value)]
        # calculate mean+std segregation index this characteristic
        mean_segreg_ind_charac = np.mean(segreg_ind_charac)
        std_segreg_ind_charac = np.std(segreg_ind_charac)
        segreg_ind.append([mean_segreg_ind_charac, std_segreg_ind_charac])

    return degree, mut_prop, cluster_coef[0], segreg_ind[0], segreg_ind[1], segreg_ind[2]


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
        Initialize the first connections using the probabilitie of linking: link_prop
        """
        g_0 = np.random.choice([0, 1], size=(self.n, self.n),
                               p=[1 - self.link_prop,
                               self.link_prop])

        # Since you can not link to yourself, the diagonal needs to be zero
        np.fill_diagonal(g_0, 0)
        return g_0

    def age_update(self):
        '''
        Not used yet
        '''
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
        '''
        This makes
        '''
        # Make room
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

        return direct_u + GAMMA * mutual_u + DELTA * indirect_u - d_i ** ALPHA * C

    def step(self):
        """ Randomly selects an agent i to revise their link with another random
        agent j. Returns the updated adjacency matrix """

        # Add noise and shuffle indexes
        eps = np.random.normal(scale=SIGMA, size=self.n*2)
        np.random.shuffle(self.indexes)

        for i in self.indexes:
            # Choose new connection
            r1 = i
            while r1 == i:
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

            if t > n_zeros_conv and stop_rule(self.zero_sequence, t):
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
                        if sex[j] == 1:
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


def true_data_make(models):
    g_matrix = pickle.load(open(r"C:\Users\FlorisFok\Downloads\g_list.pkl", 'rb'))
    big_x = pickle.load(open(r"C:\Users\FlorisFok\Downloads\x_list.pkl", 'rb'))

    true_data = {'av_degree' : [], 'clustering_coefficient' : [], 'mut_prop' : [],
                 'segreg_ind0' : [], 'segreg_ind1' : [], 'segreg_ind2' : []}
    _string = ['av_degree', 'mut_prop', 'clustering_coefficient', 'segreg_ind0', 'segreg_ind1', 'segreg_ind2']

    for num in models:
        g = g_matrix[num]
        X = big_x[num]

        output2 = analyse_network(g, X)
        for s, o in zip(_string, output2):
            try:
                true_data[s].append(o[0])
            except:
                true_data[s].append(o)

    for s in _string:
        true_data[s] = avg(true_data[s], len(models))

    return true_data

def find_models(maxi, mini):
    big_x = pickle.load(open(r"C:\Users\FlorisFok\Downloads\x_list.pkl", 'rb'))

    models = []
    for n, x in enumerate(big_x):
        if len(x) > mini and len(x) < maxi:
            models.append(n)

    return models


def main(settings, schools):
    '''

    :param settings: list of settings (=DELTA, GAMMA, C, SIGMA, B1, B2, B3)
    :param schools:  lsit of schools based on index number
    :return: The outcome of the analyse function in a dict, ordered the same way as schools.
    '''
    global DELTA, GAMMA, C, B1, B2, B3, SIGMA, ALPHA, MIN_PROP, pos_link, MINIMAL, MAX_FRIEND, zeros, n_zeros_conv

    # Plot info
    analyse_data = {'av_degree':[], 'clustering_coefficient':[], 'mut_prop':[],
                    'segreg_ind0':[], 'segreg_ind1':[], 'segreg_ind2':[]}

    _string = ['av_degree', 'mut_prop', 'clustering_coefficient', 'segreg_ind0', 'segreg_ind1', 'segreg_ind2']

    # True
    g_matrix = pickle.load(open(r"C:\Users\FlorisFok\Downloads\g_list.pkl", 'rb'))
    big_x = pickle.load(open(r"C:\Users\FlorisFok\Downloads\x_list.pkl", 'rb'))

    DELTA, GAMMA, C, SIGMA, B1, B2, B3 = settings

    # Constants
    MAX_FRIEND = 5  # given by the questionair
    MIN_PROP = 10  # Useful for steering the probabilities
    pos_link = 0.1  # Possibility to form a link on the innitial matrix
    max_iterations = 5000
    ALPHA = 2

    # Convergenge
    n_zeros_conv = 3
    zeros = np.zeros(n_zeros_conv)

    # Iterate over the given schools
    for school in schools:
        X = big_x[school]
        n_agents = len(X['sex'])

        # Make model and connection matrix
        g = ConnectionMatrix(n_agents, pos_link)
        M = Model(g, n_agents, X, [])

        # Run and rank the model
        M.run(max_iterations, 0)
        M.rank()  # chooses best 5 female and male friends

        # Collect output data
        output = analyse_network(M.g.g, X)
        for s, o in zip(_string, output):
            try:
                analyse_data[s].append(o[0])
            except:
                analyse_data[s].append(o)

    return analyse_data

if __name__ == "__main__":
    main()
