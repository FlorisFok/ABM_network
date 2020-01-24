
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import sys
import pickle
import os
import pickle

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
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


def ana_dyads(connectivity):
    """
        Calculate proportion symmetric dyads (PTCMUT), asymmetric dyads (PTCASY) and mutuality index (RHO2).
    """
    nodes = connectivity.shape[0]
    mutual_d = 0
    asym_d = 0
    out_degrees = []

    # count out_degree connections per individual and mutual and asymmetrix connections whole network
    for i in range(connectivity.shape[0]) :
        out_degree = 0
        for j in range(connectivity.shape[1]) :
            if connectivity[i][j] == 1 :
                out_degree += 1
                if connectivity[j][i] == 1 :
                    mutual_d += 0.5
                elif connectivity[j][i] != 1 :
                    asym_d += 1
        out_degrees.append(out_degree)

    total_d = mutual_d + asym_d

    # calculate proportion symmetric dyads (PTCMUT) and asymmetric dyads (PTCASY)
    PTCMUT = mutual_d / total_d
    PTCASY = asym_d / total_d

    # calculate mutuality index (RHO2) (according to Katz and Powell’s (1955))
    sum_out_degrees = sum(out_degrees)
    mean_out_degrees = sum_out_degrees / len(out_degrees)
    sum_squares_out = 0
    for i in range(len(out_degrees)) :
        sum_squares_out += (mean_out_degrees - out_degrees[i]) ** 2
    RHO2 = (2 * (nodes - 1) ** 2 * mutual_d - sum_out_degrees ** 2 + sum_squares_out) / (
                sum_out_degrees * (nodes - 1) ** 2 - sum_out_degrees ** 2 + sum_squares_out)

    return PTCMUT, PTCASY, RHO2


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
            print('step:', t, end='\r')

            self.g_sequence[t] = self.g.g
            self.zero_sequence[t] = conv_rule(self.g_sequence, t)

            if t > MINIMAL and stop_rule(self.zero_sequence, t):
                print('STOPPED at', t)
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
                old = self.g[i][j]

                self.g[i][j] = 0
                not_conn = self.U_of_matrix(i)

                self.g[i][j] = 1
                conn = self.U_of_matrix(i)

                self.g[i][j] = old
                value_of_con[i][j] = not_conn - conn

        print(value_of_con)

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

if __name__ == "__main__":
    # Settings
    name = [i for i in os.listdir() if 'xlsx' in i and 'settings' in i][0]
    settings = read_excel_settings(name)

    # Plot info
    plot_data_stopped = []
    plot_data_A1 = []
    plot_data_A2 = []
    plot_data_A3 = []
    plot_var = 'B'

    # True
    g_matrix = pickle.load(open(r"C:\Users\FlorisFok\Downloads\g_list.pkl", 'rb'))
    big_x = pickle.load(open(r"C:\Users\FlorisFok\Downloads\x_list.pkl", 'rb'))
    repeats = len(big_x)

    for row in range(len(settings[plot_var])):
        # CONSTANTS
        DELTA = settings['Delta'][row]  # weight placed on indirect links
        GAMMA = settings['Gamma'][row]  # weight placed on additional utility derived from a mutual link
        C = settings['C'][row]  # cost of forming and maintaining links
        B1,B2, B3, = settings['B1'][row], settings['B2'][row], settings['B3'][row]  # strength of preference for links_s
        SIGMA = settings['Sigma'][row]  # standard deviation of the shocks to utility
        ALPHA = settings['Alpha'][row]  # convexity of costs
        MIN_PROP = settings['min prop'][row]
        pos_link = settings['pos link'][row]

        n_zeros = int(settings['n zeros'][row])
        zeros = np.zeros(n_zeros)
        MINIMAL = n_zeros
        ########################

        for rerun in range(repeats):
            X = big_x[rerun]
            possible_X = [i[0] for i in list(X.groupby(['sex', 'race']))]
            n_agents = len(X['sex'])

            # Make and run model
            g = ConnectionMatrix(n_agents, pos_link)
            M = Model(g, n_agents, X, possible_X)
            stopped = M.run(int(settings['runs'][row]), 0)
            # M.rank()

            PTCMUT, PTCASY, RHO2 = ana_dyads(M.g.g)
            plot_data_stopped.append(stopped)
            plot_data_A1.append(PTCMUT)
            plot_data_A2.append(PTCASY)
            plot_data_A3.append(RHO2)
            M.plot_network()


    PTCMUT, PTCASY, RHO2 = [], [], []
    for g in g_matrix:
        a1, a2, a3 = ana_dyads(g)
        PTCMUT.append(a1)
        PTCASY.append(a2)
        RHO2.append(a3)

    plt.title(f'{plot_var} vs. Analyse')
    plt.subplot(121)
    plt.plot(settings[plot_var], avg(plot_data_A1, repeats), label='A1', color='r')
    plt.plot(settings[plot_var], avg(plot_data_A2, repeats), label='A2', color='g')
    plt.plot(settings[plot_var], avg(plot_data_A3, repeats), label='A3', color='b')

    le = len(settings[plot_var])
    plt.plot(settings[plot_var], [avg(PTCMUT, repeats)]*le, linestyle='--', label='trueA1', color='r')
    plt.plot(settings[plot_var], [avg(PTCASY, repeats)]*le, linestyle='--', label='trueA2', color='g')
    plt.plot(settings[plot_var], [avg(RHO2, repeats)]*le, linestyle='--', label='trueA3', color='b')

    plt.xlabel(plot_var)
    plt.legend()

    plt.subplot(122)
    plt.plot(settings[plot_var], avg(plot_data_stopped, repeats), label='stopped')
    plt.xlabel(plot_var)
    plt.legend()
    plt.show()
