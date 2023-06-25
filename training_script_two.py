from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.OriginalTheoreticalModelClassifier import OriginalTheoreticalModelClassifier

from scipy.io import loadmat
from Trajectory import Trajectory

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

final_lengths = list(set(sorted(lengths)))
final_lengths = final_lengths[len(final_lengths)//2:]

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

already_trained_networks = OriginalTheoreticalModelClassifier.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=OriginalTheoreticalModelClassifier.selected_hyperparameters())

print("Number of Lengths:", len(final_lengths))

for length in final_lengths:
    print("Training for length:", length)

    if len([network for network in already_trained_networks if network.trajectory_length == length]) == 0:
        classifier = OriginalTheoreticalModelClassifier(length, length, simulator=AndiDataSimulation)
        classifier.enable_early_stopping()
        classifier.enable_database_persistance()
        classifier.fit()
        classifier.save()
    else:
        print("Already trained!")

DatabaseHandler.disconnect()
