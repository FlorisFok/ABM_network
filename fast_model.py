import numpy as np
import math
import pickle

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
        :param total_time: Maximum iterations
        :param t_plot: Produce plots every t_plot periods
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
            
            # Convergence rule: Count the number of changes in g in the
            # n_zeros_conv last periods
            last_changes = [np.linalg.norm((self.g_sequence[t-k-1] - 
                self.g_sequence[t]), ord=1) for k in range(n_zeros_conv)]
    
            # stop if there were 0 changes in the last n_zeros_conv periods
            if t > n_zeros_conv and all(last_changes==np.zeros(n_zeros_conv)):
                break

            # Produce a plot and diagnostics every t_plot steps
            if t % t_plot == 0:
                print("degree:", np.sum(self.g.g))
                self.plot_network()

        return t

    def rank(self):
        """
        Rank the connections you have in order to reduce the connection matrix
        to max_friend male and female friends for each agent
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

def run(settings, X):
    '''
    :param settings: list of settings (=[DELTA, GAMMA, C, SIGMA, B1, B2, B3])
    :param schools:  lsit of schools based on index number
    :return: The outcome of the analyse function in a dict, ordered the same way as schools.
    '''
    global DELTA, GAMMA, C, B1, B2, B3, SIGMA, ALPHA, MIN_PROP, pos_link, MINIMAL, MAX_FRIEND, n_zeros_conv

    DELTA, GAMMA, C, SIGMA, B1, B2, B3 = settings

    # Constants
    MAX_FRIEND = 5  # maximum of 5 male and 5 female friends in the questionaire
    MIN_PROP = 10  # 0 means meeting similar individuals more likely,
    # high values -> approaching uniform probabilities 
    
    pos_link = 0.1  # Possibility to form a link on the initial matrix
    max_iterations = 5000
    ALPHA = 2 # makes cost quadratic in degree

    # Convergenge
    n_zeros_conv = 3 # periods of 0 change before convergence is declared

    # number of agents
    n_agents = len(X)

    # Make model and connection matrix
    g = ConnectionMatrix(n_agents, pos_link)
    M = Model(g, n_agents, X, [])

    # Run and rank the model
    M.run(max_iterations, 0)
    M.rank()  # chooses best 5 female and male friends

    return M.g.g
