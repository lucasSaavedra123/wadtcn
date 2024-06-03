from os.path import join
from os import makedirs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import statistics as st
import tqdm

from Trajectory import Trajectory
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskSingleLevelPredicter import WavenetTCNMultiTaskSingleLevelPredicter

PUBLIC_DATA_PATH = './public_data_validation_v1'
RESULT_PATH = './2nd_andi_challenge_results'
PATH_TRACK_1, PATH_TRACK_2 = './track_1', './track_2'

N_EXP = 10
N_FOVS = 30

network_of_reference = WavenetTCNMultiTaskSingleLevelPredicter(100, None, simulator=Andi2ndDataSimulation)
network_of_reference.load_as_file()

for exp in range(N_EXP):
    for fov in tqdm.tqdm(list(range(N_FOVS))):
        submission_file = join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}', f'fov_{fov}.txt')
        makedirs(join(RESULT_PATH, PATH_TRACK_2, f'exp_{exp}'), exist_ok=True)
        with open(submission_file, 'w') as f:
            dataframe_path = join(PUBLIC_DATA_PATH, PATH_TRACK_2, f'exp_{exp}', f'trajs_fov_{fov}.csv')
            df = pd.read_csv(dataframe_path)

            for idx in df.traj_idx.unique():
                df_traj = df[df.traj_idx == idx].sort_values('frame')

                trajectory = Trajectory(
                    x = df_traj['x'].tolist(),
                    y = df_traj['y'].tolist(),
                    t = df_traj['frame'].tolist(),
                    noisy=True,
                )

                network_of_reference.trajectory_length = trajectory.length
                state, alpha, d = network_of_reference.predict([trajectory])
                state, alpha, d = np.argmax(np.squeeze(state), axis=-1), np.squeeze(alpha), np.squeeze(d)

                break_points = rpt.Window(
                    model='l2',
                    width=5
                ).fit_predict(
                    np.stack([d, alpha], axis=1),pen=1
                )

                prediction_traj = [idx.astype(int)]

                last_break_point = 0
                for bp in break_points:
                    prediction_traj += [
                        10**np.mean((d[last_break_point:bp]*18)-12),
                        np.mean(alpha[last_break_point:bp]*2),
                        st.mode(state[last_break_point:bp]),
                        bp
                    ]
                    last_break_point = bp

                formatted_numbers = ','.join(map(str, prediction_traj))
                f.write(formatted_numbers + '\n')
