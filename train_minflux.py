from collections import defaultdict
import glob
import pandas as pd
import numpy as np
import tqdm
import json
import matplotlib.pyplot as plt

from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from DataSimulation import DeepSPTDataSimulation
from Trajectory import Trajectory

TRAIN = True

network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(1000,1000,simulator=DeepSPTDataSimulation)

if TRAIN:
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
        Y = network.transform_trajectories_to_output(trajectories)
        np.save(f"X_minflux_{dataset_type}", X)
        np.save(f"Y_minflux_{dataset_type}", Y)

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

    for traj_i in range(X.shape[0]):
        aux_list = Y[traj_i,:,0].tolist()
        try:
            last_token_position = len(aux_list) - 1 - list(reversed(aux_list)).index(-10)
        except ValueError:
            last_token_position = -1
        real_states = Y[traj_i].argmax(axis=1)[last_token_position+1:]
        predicted_states = Y_predictions[traj_i].argmax(axis=1)[last_token_position+1:]
        predicted_states = delete_short_changes(predicted_states, umbral=5)

        accuracies_by_length[len(real_states)].append(np.sum(predicted_states==real_states)/len(real_states))

    lengths = np.sort(list(accuracies_by_length.keys()))
    accuracies = [np.mean(accuracies_by_length[l]) for l in lengths]

    plt.plot(lengths, accuracies)
    plt.ylim(0,1)
    plt.show()
