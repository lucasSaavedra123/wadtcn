import glob
import pandas as pd
import numpy as np
import tqdm

from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from DataSimulation import DeepSPTDataSimulation
from Trajectory import Trajectory

network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(1000,1000,simulator=DeepSPTDataSimulation)

def transform_cache_file_chuck_files(cache_files, dataset_type):
    trajectories = []
    for cache_i, cache_file_path in enumerate(cache_files):
        cache_dataframe = pd.read_csv(cache_file_path)

        for trajectory_id in tqdm.tqdm(cache_dataframe['id'].unique()):
            trajectory_dataframe = cache_dataframe[cache_dataframe['id'] == trajectory_id]
            trajectory_dataframe = trajectory_dataframe.sort_index()

            trajectories.append(Trajectory(
                x=trajectory_dataframe['x'].tolist(),
                y=trajectory_dataframe['y'].tolist(),
                noisy=True,
                info={'state_t':trajectory_dataframe['state_t'].tolist()}
            ))

    X = network.transform_trajectories_to_input(trajectories)
    Y = network.transform_trajectories_to_input(trajectories)
    np.save(f"X_minflux_{dataset_type}", X)
    np.save(f"Y_minflux_{dataset_type}", Y)

transform_cache_file_chuck_files(glob.glob('*train*_segmentated_trajectories.cache'), 'train')
transform_cache_file_chuck_files(glob.glob('*val*_segmentated_trajectories.cache'), 'val')

network.fit()
network.save_as_file('wavenet_minflux.h5')
