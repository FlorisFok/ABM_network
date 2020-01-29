
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import sys
import pickle
import numpy

fig = plt.figure()
colors = ['#FFFFFF', '#FF0000', '#00FF00', '#0000FF',
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

    def __init__(self, g, n, shares, pos_X):
        self.n = n
        self.shares = shares
        self.pos_X = pos_X
        self.g = g

        self.X = self.make_X()
        self.U = self.make_pre_cal_U()
        self.P = self.make_prop()

        self.indexes = list(range(n))
        self.g_sequence = None
        self.zero_sequence = None

    def make_X(self):
        """
        Create X matrix
        """
        x = []
        for i, share in enumerate(self.shares):
            x += int(share * self.n) * [self.pos_X[i]]

        x += (self.n - len(x)) * [self.pos_X[i]]
        return np.array(x)

    def make_prop(self):
        # Setup probability matrix
        prop = np.zeros((self.n, self.n))

        # Loop over the person and their peers
        for i, person in enumerate(self.X):
            for j, other in enumerate(self.X):
                if i == j:
                    prop[i, j] = 0
                else:
                    prop[i, j] = np.dot(person, other) + MIN_PROP

            # Normalize
            prop[i, :] = prop[i, :] / np.sum(prop[i, :])

        return prop

    def make_pre_cal_U(self):
        """ Make the U matrix for the entire system
        """
        # Setup U
        pre_cal_u = np.zeros((self.n, self.n))

        # Fill U
        for i in range(self.n):
            for j in range(self.n):
                pre_cal_u[i, j] = math.exp(-B * np.linalg.norm((self.X[i] - self.X[j]), ord=1))

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

        # Add node colors according to X
        color_map = []
        for i in range(self.n):
            for j in range(len(self.pos_X)):
                if np.all(self.X[i] == self.pos_X[j]):
                    color_map.append(colors[j])

        fig.clear()
        nx.draw(gr, node_color=color_map, with_labels=True, node_size=500)

        if not final:
            plt.pause(1)
        else:
            plt.pause(100)

    def save2pickle(self, pickle_name):
        """
        Saves data from the simulation to a pickle file.
        :param pickle_name: Name of pickle file
        """
        # individuals_friendships_utilities = [self.X, self.g.g, self.U]
        # pickle.dump(individuals_friendships_utilities, open(pickle_name, "wb"))

    def run(self, total_time, t_plot=0):
        """
        Run the ABM simulation
        :param total_time: Number of environment evaluations
        :param t_plot: Number of times the intermediate results need to be plotted
        :return: None (use plot to show results or save to save them)
        """
        if t_plot == 0:
            t_plot = total_time

        self.g_sequence = np.zeros((total_time, self.n, self.n), dtype=numpy.int8)
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
            settings_dict[col] = float(column[0])

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

def main(settings):
    # CONSTANTS
    global DELTA, GAMMA, C, B, SIGMA, ALPHA, MIN_PROP, MINIMAL

    DELTA = settings['Delta']  # weight placed on indirect links
    GAMMA = settings['Gamma']  # weight placed on additional utility derived from a mutual link
    C = settings['C']  # cost of forming and maintaining links
    B = settings['B']  # strength of preference for links to similar agents
    SIGMA = settings['Sigma']  # standard deviation of the shocks to utility
    ALPHA = settings['Alpha']  # convexity of costs
    MIN_PROP = settings['min prop']

    n_zeros = int(settings['n zeros'])
    zeros = np.zeros(n_zeros)
    MINIMAL = n_zeros
    ########################

    # Possible relations
    possible_X = np.array(settings['pos X'])

    # Equal shares
    shares = [1 / len(possible_X)] * len(possible_X)

    # Check if there are enough colors
    if len(possible_X) > len(colors) :
        print("Amount of colors is less then possible X's")
        sys.exit()

    # Initialize arguments
    pos_link = settings['pos link']
    n_agents = int(settings['n agents'])

    # Make and run model
    g = ConnectionMatrix(n_agents, pos_link)
    M = Model(g, n_agents, shares, possible_X)
    M.run(int(settings['runs']), 0)
    # M.rank()

    # Save result
    # save the individuals_friendships
    # M.save2pickle("individuals_friendships_utilities.p")
    M.plot_network(final=True)
    fig.savefig('result.png')

if __name__ == "__main__":
    settings = {
        'Delta': 0.3,
        'Gamma': 0.5,
        'C': 0.2,
        'B': 0.3,
        'Sigma': 0.01,
        'Alpha': 2.1,
        'min prop': 10,
        'n zeros': 3,
        'pos link': 0.1,
        'n agents': 2000,
        'runs': 5000,
        'pos X': [[1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 2, 0],
                  [1, 0, 2], [0, 1, 2], [2, 2, 1], [2, 2, 0],
                  [2, 0, 2], [0, 2, 2]]
    }
    main(settings)
