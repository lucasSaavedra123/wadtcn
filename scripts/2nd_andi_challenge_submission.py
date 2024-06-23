from os.path import join
from os import makedirs
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import statistics as st
import tqdm

from Trajectory import Trajectory
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter

PUBLIC_DATA_PATH = './public_data_validation_v1'
RESULT_PATH = './2nd_andi_challenge_results'
PATH_TRACK_1, PATH_TRACK_2 = './track_1', './track_2'

N_EXP = 10
N_FOVS = 30

info_field_to_network = {
    'alpha_t': WavenetTCNSingleLevelAlphaPredicter(100, None, simulator=Andi2ndDataSimulation),
    'd_t': WavenetTCNSingleLevelDiffusionCoefficientPredicter(100, None, simulator=Andi2ndDataSimulation),
    'state_t': WavenetTCNMultiTaskClassifierSingleLevelPredicter(100, None, simulator=Andi2ndDataSimulation),
}

for field in info_field_to_network:
    info_field_to_network[field].load_as_file()

#All trajectories are extracted, stored file information and prepared for further inference
trajectories = []
print("Loading trajectories...")
for exp in tqdm.tqdm(list(range(N_EXP))):
    for fov in range(N_FOVS):
        submission_file = join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}', f'fov_{fov}.txt')
        makedirs(join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}'), exist_ok=True)
        with open(submission_file, 'w') as f:
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
            t.info[field] = prediction[t_i].flatten()*2 if field == 'alpha_t' else prediction[t_i].flatten()

#Pointwise predictions are converted into segments and results are saved
for exp in tqdm.tqdm(list(range(N_EXP))):
    for fov in range(N_FOVS):
        submission_file = join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}', f'fov_{fov}.txt')
        makedirs(join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}'), exist_ok=True)
        with open(submission_file, 'w') as f:
            exp_and_fov_trajectories = [t for t in trajectories if t.info['exp']==exp and t.info['fov']==fov]
            """
            Here, breakpoints should be identified. By now, fake breakpoints are saved
            """
            result = rpt.Pelt(model="l1").fit(t.info[field]).predict(pen=1)

            #fig, ax = plt.subplots(2,1)

            #for index in result:
            #    ax[0].axvline(index)

            #ax[0].plot(t.info[field])
            #ax[1].plot(np.diff(t.info[field]))
            #ax[0].set_ylim([0,2])
            #ax[1].set_ylim([-1,1])
            #plt.show()

            for trajectory in exp_and_fov_trajectories:
                prediction_traj = [int(trajectory.info['idx'])]
                break_points = [int(0.75 * trajectory.length), trajectory.length]
                last_break_point = 0
                for bp in break_points:
                    prediction_traj += [
                        np.mean(trajectory.info['d_t'][last_break_point:bp]),
                        np.mean(trajectory.info['alpha_t'][last_break_point:bp]),
                        st.mode(trajectory.info['state_t'][last_break_point:bp]),
                        bp
                    ]
                    last_break_point = bp

                formatted_numbers = ','.join(map(str, prediction_traj))
                f.write(formatted_numbers + '\n')
