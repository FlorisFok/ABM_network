
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
import sys

fig = plt.figure()
colors = ['#FFFFFF', '#FF0000','#00FF00', '#0000FF',
          '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0',
          '#808080', '#800000',	'#808000', '#008000',
          '#800080', '#008080', '#000080']


class Model():

    def __init__(self, n, p_link_0, shares, pos_X):
        self.n = n
        self.link_prop = p_link_0
        self.shares = shares
        self.pos_X = pos_X

        self.make_X()
        self.make_g_0()
        self.make_pre_cal_U()

        self.indexes = list(range(n))

    def make_g_0(self):
        '''
        Initialize the first connections
        '''
        g_0 = np.random.choice([0, 1], size=(n, n),
                               p=[1 - self.link_prop, self.link_prop])

        np.fill_diagonal(g_0, 0)
        self.g = g_0

    def make_X(self):
        '''
        Create X matrix
        '''
        self.X = []
        for i, share in enumerate(self.shares):
            self.X += int(share * n) * [self.pos_X[i]]

        self.X += (n - len(self.X)) * [self.pos_X[i]]
        self.X = np.array(self.X)


    def make_pre_cal_U(self):
        """ Make the U matrix for the entire system
        """
        # Setup
        n = self.n
        X = self.X
        pre_cal_u = np.zeros((n, n))

        # Fill U
        for i in range(n):
            for j in range(n):
                pre_cal_u[i, j] = math.exp(-B*np.linalg.norm((X[i] - X[j]), ord=1))

        self.U = pre_cal_u


    def U_of_matrix(self, i) :
        """ Returns the full utility of agent i given the current network structure
        g and the matrix of characteristics X """

        # degree, connection gain and cost calculations
        d_i = self.g[i].sum()
        direct_u = np.sum(self.g[i] * self.U[i])
        mutual_u = np.sum(self.g[i] * self.g.T[i] * self.U[i])

        # indirect connection gain
        a = (self.g.T.dot(self.g[i, :]) * self.U)[i]
        a[i] = 0
        indirect_u = np.sum(a)

        return direct_u + GAMMA * mutual_u + DELTA * indirect_u - cost(d_i)


    def step(self):
        """ Randomly selects an agent i to revise their link with another random
        agent j. Returns the updated adjacency matrix """

        # Add noise and shuffle indexes
        eps = np.random.normal(scale=SIGMA, size=n*2)
        np.random.shuffle(self.indexes)

        for i in self.indexes:
            # Choose new connection
            r1 = i
            while r1 == i:
                r1 = np.random.choice(self.indexes)

            # find value for new connection and removed connection
            self.g[i, r1] = 0
            U_without = self.U_of_matrix(i) + eps[i]

            self.g[i, r1] = 1
            U_with = self.U_of_matrix(i) + eps[-i]

            # Evaluate better option
            if U_without > U_with:
                self.g[i, r1] = 0
            else:
                self.g[i, r1] = 1

    def plot_network(self, final=False) :
        """ Uses networkX to plot the directed network g """
        rows, cols = np.where(self.g == 1)

        # MAke the network
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_nodes_from(range(self.n))
        gr.add_edges_from(edges)

        # Add node colors according to X
        color_map = []
        for i in range(self.n) :
            for j in range(len(self.pos_X)):
                if np.all(self.X[i] == self.pos_X[j]) :
                    color_map.append(colors[j])

        fig.clear()
        nx.draw(gr, node_color=color_map, with_labels=True, node_size=500)

        if not final:
            plt.pause(1)
        else:
            plt.pause(100)

    def run(self, T, t_plot):

        self.g_sequence = np.zeros((T, self.n, self.n))
        self.g_sequence[0] = self.g

        self.zero_sequence = np.zeros(T)
        self.zero_sequence[0] = 1.0

        for t in range(1, T):
            # Perform a step and attach the new network
            self.step()
            print('step:', t, end='\r')

            self.g_sequence[t] = self.g
            self.zero_sequence[t] = conv_rule(self.g_sequence, t)

            try :
                if t > MINIMAL and stop_rule(self.zero_sequence, t):
                    print('STOPPED at', t)
                    break
            except :
                pass

            # Produce a plot and diagnostics every t_plot steps
            if t  % t_plot == 0 :
                print("degree:", np.sum(self.g))
                # self.plot_network()

    def rank(self):
        '''
        Rank the connections you have
        :return:
        '''

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

if __name__ == "__main__":
    # CONSTANTS
    DELTA = 0.3  # weight placed on indirect links
    GAMMA = 0.5  # weight placed on additional utility derived from a mutual link
    C = 0.2  # cost of forming and maintaining links
    B = 0.3  # strength of preference for links to similar agents
    SIGMA = 0.01  # standard deviation of the shocks to utility
    ALPHA = 2.1  # convexity of costs

    ## Cost function #######
    def cost(d_i):
        return d_i ** ALPHA * C

    ## conv_rule ###########
    def conv_rule(g_sequence, t):
        return np.linalg.norm((g_sequence[t-1] - g_sequence[t]), ord=1)
        # return (np.sum(g_sequence[t] - g_sequence[t - 1]))

    ## STOP RULE ###########
    def stop_rule(zero_sequence, t):
        return zero_sequence[t - n_zeros :t].any() == zeros.any()

    n_zeros = 3
    zeros = np.zeros(n_zeros)
    MINIMAL = n_zeros
    ########################

    # Possible relations
    # possible_X = [[1, 0], [0, 1], [1, 1]]
    possible_X = [[1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 2, 0],
                  [1, 0, 2], [0, 1, 2], [2, 2, 1], [2, 2, 0],
                  [2, 0, 2], [0, 2, 2]]

    # Equal shares
    shares = [1/len(possible_X)] * len(possible_X)

    # Check if there are enough colors
    if len(possible_X)  > len(colors):
        print("Amount of colors is less then possible X's")
        sys.exit()

    # Initialize arguments
    pos_link = 0.1
    n = 100  # Number of agents

    # Make and run model
    M = Model(n, pos_link, shares, possible_X)
    M.run(500, 10)
    # M.rank()

    # Save result
    M.plot_network(final=True)
    fig.savefig('result.png')