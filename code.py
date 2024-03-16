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

from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

def generate_trajectories(limit):
    EXPERIMENTS = np.arange(5).repeat(2)
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

    t = []

    #for idx, (i, fix_exp) in tqdm.tqdm(enumerate(zip(EXPERIMENTS, exp_dic))):

    with tqdm.tqdm(total=limit) as pbar:
        while len(t) < limit:
            idx = np.random.randint(0, len(EXPERIMENTS))
            i = EXPERIMENTS[idx]
            fix_exp = exp_dic[idx]

            dic = _get_dic_andi2(i+1)
            dic['T'] = 500
            dic['N'] = 100

            for key in fix_exp:
                dic[key] = fix_exp[key]

            old_t = len(t)
            for _ in range(NUM_FOVS):
                trajs, labels = datasets_phenom().create_dataset(dics = dic)
                t += [ti for ti in Trajectory.from_datasets_phenom(trajs, labels) if len(np.unique(ti.info['alpha_t'])) != 1 and 0 not in ti.info['alpha_t']]

            pbar.update(len(t) - old_t)

    return t

network = WavenetTCNWithLSTMHurstExponentSingleLevelPredicter(500, 500, simulator=CustomDataSimulation)
#network = WavenetTCNWithLSTMModelSingleLevelPredicter(200, 200, simulator=CustomDataSimulation)

t = generate_trajectories(100_000)
X_train, Y_train = network.transform_trajectories_to_input(t), network.transform_trajectories_to_output(t)

t = generate_trajectories(12_500)
X_val, Y_val = network.transform_trajectories_to_input(t), network.transform_trajectories_to_output(t)


network.build_network()

network.architecture.summary()
network.architecture.fit(
    X_train, Y_train,
    epochs=10,
    validation_data=[X_val, Y_val],
    shuffle=True
)

network.save_as_file()
#network.load_as_file()

result = network.predict(t)
#result = np.argmax(result,axis=2)
idxs = np.arange(0,len(t), 1)
np.random.shuffle(idxs)

for i in idxs:
    ti = t[i]
    plt.plot(ti.info['alpha_t'])
    plt.plot(result[i, :]*2)
    plt.ylim([0,2])
    plt.show()

"""
result = network.predict(t)
result = np.argmax(result,axis=2)

true = []
pred = []

for i, ti in enumerate(t):
    true += ti.info['state_t']
    pred += result[i,:].tolist()

print(f1_score(true, pred, average='micro'))

confusion_mat = confusion_matrix(y_true=true, y_pred=pred)
confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

labels = ["0", "1", "2", "3"]

confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
sns.set(font_scale=1.5)
color_map = sns.color_palette(palette="Blues", n_colors=7)
sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

plt.title(f'Confusion Matrix')
plt.rcParams.update({'font.size': 15})
plt.ylabel("Ground truth", fontsize=15)
plt.xlabel("Predicted label", fontsize=15)
plt.show()
"""