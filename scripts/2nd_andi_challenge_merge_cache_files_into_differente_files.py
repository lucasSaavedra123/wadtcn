import glob
import pandas as pd
import os
import tqdm
import numpy as np

from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskSingleLevelPredicter import WavenetTCNMultiTaskSingleLevelPredicter
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from Trajectory import Trajectory


classifier_network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(None, None, simulator=Andi2ndDataSimulation)
regression_network = WavenetTCNMultiTaskSingleLevelPredicter(None, None, simulator=Andi2ndDataSimulation)

DEST_DIRECTORY = '2ndAndiTrajectories'
os.makedirs(DEST_DIRECTORY, exist_ok=True)
trajectory_counter = 0

cache_files = glob.glob('*train*.cache')

for cache_i, cache_file_path in enumerate(cache_files):
    cache_dataframe = pd.read_csv(cache_file_path)
    
    for trajectory_id in tqdm.tqdm(cache_dataframe['id'].unique()):
        trajectory_dataframe = cache_dataframe[cache_dataframe['id'] == trajectory_id]
        trajectory_dataframe = trajectory_dataframe.sort_values('t')

        trajectory = Trajectory(
            x = trajectory_dataframe['x_noisy'].tolist(),
            y = trajectory_dataframe['y_noisy'].tolist(),
            t = trajectory_dataframe['t'].tolist(),
            info={
                'd_t': trajectory_dataframe['d_t'].tolist(),
                'alpha_t': trajectory_dataframe['alpha_t'].tolist(),
                'state_t': trajectory_dataframe['state_t'].tolist()
            },
            noisy=True
        )

        X_train, Y_train = classifier_network.transform_trajectories_to_input([trajectory]), classifier_network.transform_trajectories_to_output([trajectory])
        
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_X.npy'), X_train)
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_Y0.npy'), Y_train)

        _, Y_train = regression_network.transform_trajectories_to_input([trajectory]), regression_network.transform_trajectories_to_output([trajectory])

        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_Y1.npy'), Y_train[0])
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_Y2.npy'), Y_train[1])

        trajectory_counter += 1
