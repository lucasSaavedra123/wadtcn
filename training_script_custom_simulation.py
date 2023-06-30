from DatabaseHandler import DatabaseHandler
from DataSimulation import CustomDataSimulation
from TheoreticalModels import ALL_MODELS
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.LSTMTheoreticalModelClassifier import LSTMTheoreticalModelClassifier
from PredictiveModel.LSTMAnomalousExponentPredicter import LSTMAnomalousExponentPredicter
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.OriginalTheoreticalModelClassifier import OriginalTheoreticalModelClassifier

from scipy.io import loadmat
from Trajectory import Trajectory

from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME

import numpy as np


mat_data = loadmat('all_tracks_thunder_localizer.mat')
# Orden en la struct [BTX|mAb] [CDx|Control|CDx-Chol]
dataset = []
# Add each label and condition to the dataset
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][0][0]})
dataset.append({'label': 'BTX',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][0][1]})
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][0][2]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][1][0]})
dataset.append({'label': 'mAb',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][1][1]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][1][2]})

lengths = []

for data in dataset:
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])
    for trajectory in trajectories:
        if not trajectory.is_immobile(1.8) and trajectory.length >= 25:
            lengths.append(trajectory.length)

final_lengths = set(sorted(lengths))

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())

print("Number of Lengths:", len(final_lengths))

for length in final_lengths:
    print("Training for length:", length)

    networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

    if len(networks_of_length) == 0:
        classifier = WaveNetTCNTheoreticalModelClassifier(length, EXPERIMENT_TIME_FRAME_BY_FRAME * length, simulator=CustomDataSimulation)
        classifier.enable_early_stopping()
        classifier.enable_database_persistance()
        classifier.fit()
        classifier.save()
    else:
        print("Already trained!")

DatabaseHandler.disconnect()
