import glob
import pandas as pd
import os
import tqdm
import numpy as np

from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from Trajectory import Trajectory


classifier_network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(None, None, simulator=Andi2ndDataSimulation)
d_regression_network = WavenetTCNSingleLevelDiffusionCoefficientPredicter(None, None, simulator=Andi2ndDataSimulation)
alpha_regression_network = WavenetTCNSingleLevelAlphaPredicter(None, None, simulator=Andi2ndDataSimulation)

DEST_DIRECTORY = '2ndAndiTrajectories'
os.makedirs(DEST_DIRECTORY, exist_ok=True)

classification_cache_files = glob.glob('*train*classification.cache')
trajectory_counter = 0
for cache_i, cache_file_path in enumerate(classification_cache_files):
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
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_X_classifier.npy'), X_train)
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_Y_classifier.npy'), Y_train)
        trajectory_counter += 1

classification_cache_files = glob.glob('*train*regression.cache')
trajectory_counter = 0
for cache_i, cache_file_path in enumerate(classification_cache_files):
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

        X_train, Y_train = d_regression_network.transform_trajectories_to_input([trajectory]), d_regression_network.transform_trajectories_to_output([trajectory])
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_X_D_regression.npy'), X_train)
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_Y_D_regression.npy'), Y_train)

        X_train, Y_train = alpha_regression_network.transform_trajectories_to_input([trajectory]), alpha_regression_network.transform_trajectories_to_output([trajectory])
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_X_A_regression.npy'), X_train)
        np.save(os.path.join(DEST_DIRECTORY, f'{trajectory_counter}_Y_A_regression.npy'), Y_train)

        trajectory_counter += 1
