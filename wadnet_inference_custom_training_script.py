import os
import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.backend import clear_session

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from DataSimulation import CustomDataSimulation
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from TheoreticalModels import ALL_SUB_MODELS
from CONSTANTS import IMMOBILE_THRESHOLD, EXPERIMENT_TIME_FRAME_BY_FRAME

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

if not os.path.exists('lengths_cache.txt'):
    lengths = np.sort(np.unique([int(trajectory.length) for trajectory in Trajectory.objects() if (not trajectory.is_immobile(IMMOBILE_THRESHOLD)) and trajectory.length >= 25]))

    with open('lengths_cache.txt', 'w') as file:
        file.write('\n'.join(str(length) for length in lengths))
else:
    with open('lengths_cache.txt', 'r') as file:
        lengths = [int(line.strip()) for line in file.readlines()]

reference_architectures = {
    WaveNetTCNFBMModelClassifier: None,
    WaveNetTCNSBMModelClassifier: None,
    WavenetTCNWithLSTMHurstExponentPredicter: {},
}

for index, length in tqdm.tqdm(enumerate(lengths)):
    clear_session()
    for network_class in [WaveNetTCNFBMModelClassifier, WaveNetTCNSBMModelClassifier, WavenetTCNWithLSTMHurstExponentPredicter]:

        if network_class == WavenetTCNWithLSTMHurstExponentPredicter:

            for class_model in ALL_SUB_MODELS:
                already_trained_networks = network_class.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters(class_model.STRING_LABEL))

                networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length and network.extra_parameters['model'] == class_model.STRING_LABEL]

                if len(networks_of_length) == 0:
                    classifier = network_class(length, length * EXPERIMENT_TIME_FRAME_BY_FRAME, simulator=CustomDataSimulation, model=class_model.STRING_LABEL)

                    if index == 0:
                        reference_architectures[network_class][class_model.STRING_LABEL] = classifier
                    else:
                        classifier.set_wadnet_tcn_encoder(reference_architectures[network_class][class_model.STRING_LABEL], -3)

                    classifier.enable_early_stopping()
                    classifier.enable_database_persistance()
                    classifier.fit()
                    classifier.save()
                else:
                    assert len(networks_of_length) == 1
                    classifier = networks_of_length[0]

                    if index == 0:
                        reference_architectures[network_class][class_model.STRING_LABEL] = classifier
                    else:
                        classifier.set_wadnet_tcn_encoder(reference_architectures[network_class][class_model.STRING_LABEL], -3)

                    classifier.enable_database_persistance()
                    classifier.load_as_file()

        else:
            already_trained_networks = network_class.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters())

            networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

            if len(networks_of_length) == 0:
                classifier = network_class(length, length * EXPERIMENT_TIME_FRAME_BY_FRAME, simulator=CustomDataSimulation)

                if index == 0:
                    reference_architectures[network_class] = classifier
                else:
                    classifier.set_wadnet_tcn_encoder(reference_architectures[network_class], -4)

                classifier.enable_early_stopping()
                classifier.enable_database_persistance()
                classifier.fit()
                classifier.save()
            else:
                assert len(networks_of_length) == 1
                classifier = networks_of_length[0]

                if index == 0:
                    reference_architectures[network_class] = classifier
                else:
                    classifier.set_wadnet_tcn_encoder(reference_architectures[network_class], -4)

                classifier.enable_database_persistance()
                classifier.load_as_file()


DatabaseHandler.disconnect()