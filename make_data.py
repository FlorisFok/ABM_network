import numpy as np
import model
import pickle


settings =  [0.05, 0.65, 0.175, 0.035, 0.1, 0.1, 0.2]
_string = ['av_degree', 'mut_prop', 'clustering_coefficient', 'segreg_ind2', 'segreg_ind1', 'segreg_ind0']

extraruns = {'av_degree':[], 'clustering_coefficient':[], 'mut_prop':[],
                    'segreg_ind0':[], 'segreg_ind1':[], 'segreg_ind2':[]}

# 100 normal runs
data = model.main(settings, [4] * 100)
pickle.dump(data, open('100xschool4.pkl', 'wb'))

# 100 runs with extra X column
for i in range(100):
    print('--', i, end='\r')
    big_x = pickle.load(open(r"C:\Users\FlorisFok\Downloads\x_list.pkl", 'rb'))
    X = big_x[4]
    new = np.random.randint(0, 5, (len(X),))
    X['new'] = new
    pickle.dump([X], open('custom_x.pkl', 'wb'))
    new_data = model.main(settings, [0])
    for s in _string:
        extraruns[s].append(new_data[s][0])

pickle.dump(extraruns, open('100xschool4extra.pkl', 'wb'))