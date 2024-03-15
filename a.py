# Generic libraries
import pandas as pd
import numpy as np
np.random.seed(0)

import tqdm

# Functions needed from andi_datasets
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2, _defaults_andi2
from andi_datasets.utils_challenge import load_file_to_df, error_SingleTraj_dataset, file_nonOverlap_reOrg

from andi_datasets.datasets_phenom import datasets_phenom
import matplotlib.pyplot as plt
from Trajectory import Trajectory
from DataSimulation import AndiDataSimulation, CustomDataSimulation
from PredictiveModel.WavenetTCNWithLSTMHurstExponentSingleLevelPredicter import WavenetTCNWithLSTMHurstExponentSingleLevelPredicter
from PredictiveModel.WavenetTCNWithLSTMModelSingleLevelPredicter import WavenetTCNWithLSTMModelSingleLevelPredicter


EXPERIMENTS = np.arange(5).repeat(4)
NUM_FOVS = 30

# We create a list of dictionaries with the properties of each experiment
exp_dic = [None]*len(EXPERIMENTS)


##### SINGLE STATE #####
exp_dic[0] = {'Ds': [1, 0.01], 'alphas' : [0.5, 0.01]}
exp_dic[1] = {'Ds': [0.1, 0.01], 'alphas' : [1.9, 0.01]}

##### MULTI STATE #####
exp_dic[2] = {'Ds': np.array([[1, 0.01], [0.05, 0.01]]),
              'alphas' : np.array([[1.5, 0.01], [0.5, 0.01]]),
              'M': np.array([[0.99, 0.01],[0.01, 0.99]])
             }
exp_dic[3] = {'Ds': np.array([[1, 0.01], [0.5, 0.01], [0.01, 0.01]]),
              'alphas' : np.array([[1.5, 0.01], [0.5, 0.01], [0.75, 0.01]]),
              'M': np.array([[0.98, 0.01, 0.01],[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])
             }

##### IMMOBILE TRAPS #####
exp_dic[4] = {'Ds': [1, 0.01], 'alphas' : [0.8, 0.01],
              'Pu': 0.01, 'Pb': 1,
              'Nt': 300, 'r': 0.6}
exp_dic[5] = {'Ds': [1, 0.01], 'alphas' : [1.5, 0.01],
              'Pu': 0.05, 'Pb': 1,
              'Nt': 150, 'r': 1}

##### DIMERIZATION #####
exp_dic[6] = {'Ds': np.array([[1, 0.01], [1, 0.01]]), 'alphas' : np.array([[1.2, 0.01], [0.8, 0.01]]),
              'Pu': 0.01, 'Pb': 1,  'N': 100, 'r': 0.6}
exp_dic[7] = {'Ds': np.array([[1, 0.01], [3, 0.01]]), 'alphas' : np.array([[1.2, 0.01], [0.5, 0.01]]),
              'Pu': 0.01, 'Pb': 1,  'N': 80, 'r': 1}

##### CONFINEMENT #####
exp_dic[8] = {'Ds': np.array([[1, 0.01], [1, 0.01]]), 'alphas' : np.array([[0.8, 0.01], [0.4, 0.01]]),
              'Nc': 30, 'trans': 0.1, 'r': 5}
exp_dic[9] = {'Ds': np.array([[1, 0.01], [0.1, 0.01]]), 'alphas' : np.array([[1, 0.01], [1, 0.01]]),
              'Nc': 30, 'trans': 0, 'r': 10}

dics = []

#network = WavenetTCNWithLSTMHurstExponentSingleLevelPredicter(200, 200, simulator=AndiDataSimulation)
network = WavenetTCNWithLSTMModelSingleLevelPredicter(200, 200, simulator=CustomDataSimulation)

t = []

for idx, (i, fix_exp) in tqdm.tqdm(enumerate(zip(EXPERIMENTS, exp_dic))):

    dic = _get_dic_andi2(i+1)
    dic['T'] = 200
    dic['N'] = 200

    for key in fix_exp:
        dic[key] = fix_exp[key]

    for _ in range(NUM_FOVS):
        trajs, labels = datasets_phenom().create_dataset(dics = dic)
        t += Trajectory.from_datasets_phenom(trajs, labels)

t = [ti for ti in t if len(np.unique(ti.info['state_t'])) != 1]

print(len(t))

X_train, Y_train = network.transform_trajectories_to_input(t), network.transform_trajectories_to_output(t)

t = []

for idx, (i, fix_exp) in tqdm.tqdm(enumerate(zip(EXPERIMENTS, exp_dic))):

    dic = _get_dic_andi2(i+1)
    dic['T'] = 200
    dic['N'] = 100

    for key in fix_exp:
        dic[key] = fix_exp[key]

    for _ in range(1):
        trajs, labels = datasets_phenom().create_dataset(dics = dic)
        t += Trajectory.from_datasets_phenom(trajs, labels)

t = [ti for ti in t if len(np.unique(ti.info['state_t'])) != 1]

print(len(t))

X_val, Y_val = network.transform_trajectories_to_input(t), network.transform_trajectories_to_output(t)

network.build_network()
network.architecture.summary()

network.architecture.fit(
    X_train, Y_train,
    epochs=5,
    validation_data=[X_val, Y_val],
    shuffle=True
)

network.save_as_file()
#network.load_as_file()

result = network.predict(t)
result = np.argmax(result,axis=2)

for i, ti in enumerate(t):
    plt.plot(ti.info['state_t'])
    plt.plot(result[i,:])
    plt.ylim([-1,5])
    plt.show()
