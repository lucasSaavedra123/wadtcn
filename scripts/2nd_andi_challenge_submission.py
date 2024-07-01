from os.path import join
from os import makedirs
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import glob
import statistics as st
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from Trajectory import Trajectory
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter

PUBLIC_DATA_PATH = './public_data_challenge_v0'
RESULT_PATH = './2nd_andi_challenge_results'
PATH_TRACK_1, PATH_TRACK_2 = './track_1', './track_2'

N_EXP = 12
N_FOVS = 30

info_field_to_network = {
    'alpha_t': WavenetTCNSingleLevelAlphaPredicter(200, None, simulator=Andi2ndDataSimulation),
    'd_t': WavenetTCNSingleLevelDiffusionCoefficientPredicter(200, None, simulator=Andi2ndDataSimulation),
    'state_t': WavenetTCNMultiTaskClassifierSingleLevelPredicter(200, None, simulator=Andi2ndDataSimulation),
}

for field in info_field_to_network:
    info_field_to_network[field].load_as_file()

#All trajectories are extracted, stored file information and prepared for further inference
trajectories = []
print("Loading trajectories...")
for exp in tqdm.tqdm(list(range(N_EXP))):
    for fov in range(N_FOVS):
        dataframe_path = join(PUBLIC_DATA_PATH, PATH_TRACK_2, f'exp_{exp}', f'trajs_fov_{fov}.csv')
        df = pd.read_csv(dataframe_path)

        for idx in df.traj_idx.unique():
            df_traj = df[df.traj_idx == idx].sort_values('frame')

            trajectories.append(Trajectory(
                x = df_traj['x'].tolist(),
                y = df_traj['y'].tolist(),
                t = df_traj['frame'].tolist(),
                noisy=True,
                info={
                    'idx': idx,
                    'exp': exp,
                    'fov': fov
                }
            ))

            #plt.plot(df_traj['x'].tolist(), df_traj['y'].tolist())
        #plt.show()

print("Number of trajectories:", len(trajectories))

#Divide trajectories by length for inference acceleration
trajectories_by_length = defaultdict(lambda: [])
for trajectory in trajectories:
    trajectories_by_length[trajectory.length].append(trajectory)
print("Number of lengths:", len(trajectories_by_length.keys()))

#Inference and results stored in trajectories
print("Inference...")
for trajectory_length in tqdm.tqdm(trajectories_by_length):
    #Here we infere with neural networks and save results
    #By now, fake results are saved
    for field in info_field_to_network:
        prediction = info_field_to_network[field].predict(trajectories_by_length[trajectory_length]) 
        if field == 'state_t':
            prediction = np.argmax(prediction,axis=2)
        for t_i, t in enumerate(trajectories_by_length[trajectory_length]):
            flatten_input = prediction[t_i].flatten()
            assert len(flatten_input) == t.length
            if field == 'alpha_t':
                t.info[field] = flatten_input*2
            elif field == 'd_t':
                t.info[field] = 10**flatten_input
            else:
                #new_states = []
                #for frame_i in range(len(flatten_input)):
                #    if frame_i < 1:
                #        new_states.append(flatten_input[0])
                #    elif frame_i == len(flatten_input) - 1:
                #        new_states.append(flatten_input[-1])
                #    else:
                #        new_states.append(st.mode(flatten_input[frame_i-1:frame_i+2]))
                #
                #    if new_states[-1] == 0:
                #        t.info['alpha_t'][frame_i] = 0
                #        t.info['d_t'][frame_i] = 0
                #    elif new_states[-1] == 3:
                #        t.info['alpha_t'][frame_i] = 1.99
                t.info['alpha_t'][flatten_input == 0] == 0
                t.info['d_t'][flatten_input == 0] == 0
                t.info['alpha_t'][flatten_input == 3] == 2
                t.info[field] = np.array(flatten_input)
                #plt.plot(t.info[field])
                #plt.plot(new_states)
                #plt.show()
    #for t in trajectories_by_length[trajectory_length]:
    #    t.plot_andi_2(absolute_d=True, show_break_points=True)

#Pointwise predictions are converted into segments and results are saved
for exp in tqdm.tqdm(list(range(N_EXP))):
    for fov in range(N_FOVS):
        makedirs(join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}'), exist_ok=True)
        submission_file = join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}', f'fov_{fov}.txt')
        with open(submission_file, 'w') as f:
            exp_and_fov_trajectories = [t for t in trajectories if t.info['exp']==exp and t.info['fov']==fov]
            """
            Here, breakpoints are identified looking at the alpha signal
            """

            for trajectory in exp_and_fov_trajectories:
                prediction_traj = [int(trajectory.info['idx'])]
                break_points = rpt.Pelt(model="l1").fit(trajectory.info['alpha_t']).predict(pen=1)

                last_break_point = 0
                for bp in break_points:
                    state = st.mode(trajectory.info['state_t'][last_break_point:bp])

                    prediction_traj += [
                        np.mean(trajectory.info['d_t'][last_break_point:bp]),
                        np.mean(trajectory.info['alpha_t'][last_break_point:bp]),
                        state,
                        bp
                    ]

                    last_break_point = bp

                formatted_numbers = ','.join(map(str, prediction_traj))
                f.write(formatted_numbers + '\n')

#From Pointwise predictions, ensemble
for exp in range(N_EXP):
    files_path = glob.glob(join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}', 'fov_*.txt'))

    complete_info = {
        'd': [],
        'alpha': [],
        'duration': []
    }

    for file_path in files_path:
        results_file = open(file_path,'r')

        trajectories_info = [[float(d) for d in line.split(',')[1:]] for line in results_file.readlines()]

        for trajectory_info in trajectories_info:
            last_bp = 0
            for i in range(len(trajectory_info)//4):
                complete_info['d'].append(trajectory_info[i*4])
                complete_info['alpha'].append(trajectory_info[(i*4)+1])
                complete_info['duration'].append(trajectory_info[(i*4)+3] - last_bp)
                last_bp = trajectory_info[(i*4)+3]

    dataframe = pd.DataFrame(complete_info)
    dataframe['label'] = GaussianMixture(n_components=2).fit_predict(dataframe.values)
    ensemble_labels_file = join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}', 'ensemble_labels.txt')

    with open(ensemble_labels_file, 'w') as f:
        model_name = 'confinement'
        f.write(f'model: {model_name}; num_state: {2} \n')

        data = np.random.rand(5, 2)

        for label in [0,1]:
            label_dataframe = dataframe[dataframe['label'] == label]
            data[0, label] = label_dataframe['alpha'].mean()
            data[1, label] = label_dataframe['alpha'].std()
            data[2, label] = label_dataframe['d'].mean()
            data[3, label] = label_dataframe['d'].std()
            data[3, label] = label_dataframe['duration'].sum()
        
        #data[-1,:] /= data[-1,:].sum()

        # Save the data in the corresponding ensemble file
        np.savetxt(f, data, delimiter = ';')
