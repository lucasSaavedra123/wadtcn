from os.path import join
from os import makedirs
from collections import defaultdict

from IPython import embed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import statistics as st
import tqdm

from Trajectory import Trajectory
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter

PUBLIC_DATA_PATH = './public_data_validation_v1'
RESULT_PATH = './2nd_andi_challenge_results'
PATH_TRACK_1, PATH_TRACK_2 = './track_1', './track_2'

N_EXP = 10
N_FOVS = 30

#single_level_classifier = WavenetTCNMultiTaskClassifierSingleLevelPredicter(100, None, simulator=Andi2ndDataSimulation)
#single_level_classifier.build_network()
#single_level_classifier.load_as_file()

#All trajectories are extracted, stored file information and prepared for further inference
trajectories = []

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
#embed()

#Inference and results stored in trajectories
print("Inference...")
for trajectory_length in tqdm.tqdm(trajectories_by_length):
    #Here we infere with neural networks and save results
    #By now, fake results are saved

    state_fraction = int(0.75 * trajectory_length)

    for t in trajectories_by_length[trajectory_length]:
        t.info['alpha_t'] = [0.32] * state_fraction + [1.78] * (trajectory_length-state_fraction)
        t.info['d_t'] = [0.025] * state_fraction + [1.25] * (trajectory_length-state_fraction)
        t.info['state_t'] = [3] * state_fraction + [1] * (trajectory_length-state_fraction)


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
            """
            break_points = rpt.Window(
                model='l2',
                width=5
            ).fit_predict(
                np.stack([d, alpha], axis=1),pen=1
            )
            """
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
