from multiprocessing.pool import Pool
from SALib.sample import saltelli
import pickle
import numpy as np
import model

def single_pool(inp):
    print(inp[1])
    out = model.main(inp[0], [4])
    out_list = np.array([i[0] for i in out.values()])

    return inp[1], out_list

if __name__ == '__main__':
    _string = ['av_degree', 'mut_prop', 'clustering_coefficient', 'segreg_ind2', 'segreg_ind1', 'segreg_ind0']
    _var = ['Delta', 'Gamma', 'C', 'Sigma', "b_1", 'b_2', 'b_3']

    problem = {
        'num_vars': len(_var),
        'names': _var,
        'bounds': [[0, 3],
                    [0, 1],
                    [0, 3],
                    [0, 1],
                    [0, 3],
                    [0, 3],
                    [0, 3]]
    }

    param_values = saltelli.sample(problem, 500)
    Y = np.zeros([param_values.shape[0], len(_string)])

    input = list(zip(param_values, [i for i in range(len(param_values))]))
    print('total:', input[-1][1])

    p = Pool(processes=6)
    returns = p.map(single_pool, input)

    p.close()
    p.join()

    pickle.dump(returns, open('SA_sobol.p', 'wb'))
    pickle.dump(param_values, open('Par_sobol.p', 'wb'))
