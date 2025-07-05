from collections import defaultdict
import glob
import pandas as pd
import numpy as np
import tqdm
import pickle
import json
import matplotlib.pyplot as plt

from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from DataSimulation import DeepSPTDataSimulation
from CONSTANTS import *

assert FOR_MINFLUX
TRAIN = True

network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(1000,1000,simulator=DeepSPTDataSimulation)

if TRAIN:
    for i in range(2):
        print("Train dataset", i)
        DeepSPTDataSimulation().simulate_segmentated_trajectories(TRAINING_SET_SIZE_PER_EPOCH,1_000,None,True,f'train_{i}', True)

    for i in range(1):
        print("Val dataset", i)
        DeepSPTDataSimulation().simulate_segmentated_trajectories(VALIDATION_SET_SIZE_PER_EPOCH,1_000,None,True,f'val_{i}', True)

    def transform_cache_file_chuck_files(cache_files, dataset_type):
        for cache_i, cache_file_path in enumerate(cache_files):
            trajectories = []

            cache_dataframe = pd.read_csv(cache_file_path)

            for trajectory_id in tqdm.tqdm(cache_dataframe['id'].unique()):
                trajectory_dataframe = cache_dataframe[cache_dataframe['id'] == trajectory_id]
                trajectory_dataframe = trajectory_dataframe.sort_index()

                trajectories.append({
                    'x':trajectory_dataframe['x'].tolist(),
                    'y':trajectory_dataframe['y'].tolist(),
                    't':trajectory_dataframe['t'].tolist(),
                    'info':{'state_t':trajectory_dataframe['state_t'].tolist()}
                })

            with open(f'minflux_{dataset_type}_{cache_i}.json', 'w') as a_file:
                json.dump(trajectories, a_file)

    transform_cache_file_chuck_files(glob.glob('*train*_segmentated_trajectories.cache'), 'train')
    transform_cache_file_chuck_files(glob.glob('*val*_segmentated_trajectories.cache'), 'val')

    network.enable_early_stopping()
    network.fit()
    network.save_as_file('wavenet_minflux.weights.h5')

    with open("networks/wavenet_minflux.json", "w") as info_file:
        json.dump(network.history_training_info, info_file)
else:
    network.load_as_file('wavenet_minflux.weights.h5')
"""
lengths = list(range(200,1000,50))
scores = []
for length in tqdm.tqdm(lengths):
    network.trajectory_length = length
    scores.append(network.f1_score())

pd.DataFrame({'lengths':lengths, 'f1-score':scores}).to_csv("run_and_turn_scores.csv")

plt.plot(lengths,scores)
plt.show()
"""
network.plot_minflux_confusion_matrix()
