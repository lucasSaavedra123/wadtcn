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
from Trajectory import Trajectory
from CONSTANTS import *

assert FOR_MINFLUX
TRAIN = True

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    X = np.load(f"X_minflux_val.npy")
    Y = np.load(f"Y_minflux_val.npy")
    Y_predictions = network.architecture.predict(X)

    accuracies_by_length = defaultdict(list)

    def delete_short_changes(signal, umbral=5):
        signal = np.array(signal)
        result = signal.copy()
        actual = signal[0]
        initial_index = 0

        for i in range(1, len(signal)):
            if signal[i] != actual:
                duracion = i - initial_index
                if duracion <= umbral:
                    result[initial_index:i+1] = actual
                else:
                    actual = signal[i]
                    initial_index = i
        return result

    real = []
    predicted = []

    for traj_i in range(X.shape[0]):
        aux_list = Y[traj_i,:,0].tolist()
        try:
            last_token_position = len(aux_list) - 1 - list(reversed(aux_list)).index(-10)
        except ValueError:
            last_token_position = -1
        real_states = Y[traj_i].argmax(axis=1)[last_token_position+1:]
        predicted_states = Y_predictions[traj_i].argmax(axis=1)[last_token_position+1:]

        predicted_states = delete_short_changes(predicted_states, umbral=25)
        accuracies_by_length[len(real_states)].append(np.sum(predicted_states==real_states)/len(real_states))

        real.extend(real_states.tolist())
        predicted.extend(predicted_states.tolist())


    cm = confusion_matrix(real, predicted)#, labels=clf.classes_)
    cm = cm / cm.sum(axis=1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)#, display_labels=clf.classes_)
    disp.plot()
    plt.show()

    lengths = np.sort(list(accuracies_by_length.keys()))
    accuracies = [np.mean(accuracies_by_length[l]) for l in lengths]

    plt.plot(lengths, accuracies)
    plt.ylim(0,1)
    plt.show()
