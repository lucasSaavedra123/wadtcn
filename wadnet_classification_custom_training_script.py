import os

import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.backend import clear_session

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from DataSimulation import CustomDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier

from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME, IMMOBILE_THRESHOLD

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

if not os.path.exists('lengths_cache.txt'):
    lengths = np.sort(np.unique([int(trajectory.length) for trajectory in Trajectory.objects() if (not trajectory.is_immobile(IMMOBILE_THRESHOLD)) and trajectory.length >= 25]))

    with open('lengths_cache.txt', 'w') as file:
        file.write('\n'.join(str(length) for length in lengths))
else:
    with open('lengths_cache.txt', 'r') as file:
        lengths = [int(line.strip()) for line in file.readlines()]

already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(
    simulator_identifier=CustomDataSimulation.STRING_LABEL,
    trained=True,
    hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters(),
)

for index, length in tqdm.tqdm(list(enumerate(lengths))):
    print("Length:", length)
    clear_session()
    networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

    if len(networks_of_length) == 0:
        classifier = WaveNetTCNTheoreticalModelClassifier(length, length * EXPERIMENT_TIME_FRAME_BY_FRAME, simulator=CustomDataSimulation)

        if index == 0:
            reference_architecture = classifier
        else:
            classifier.set_wadnet_tcn_encoder(reference_architecture, -4)

        classifier.enable_early_stopping()
        classifier.enable_database_persistance()
        classifier.fit()
        classifier.save()
    else:
        assert len(networks_of_length) == 1
        classifier = networks_of_length[0]

        if index == 0:
            reference_architecture = classifier
        else:
            classifier.set_wadnet_tcn_encoder(reference_architecture, -4)

        classifier.enable_database_persistance()
        classifier.load_as_file()

DatabaseHandler.disconnect()
