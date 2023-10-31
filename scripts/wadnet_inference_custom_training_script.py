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

if not os.path.exists('lengths_and_durations_cache.txt'):
    lengths_durations = [(trajectory.length, round(trajectory.duration,2)) for trajectory in Trajectory.objects() if (not trajectory.is_immobile(IMMOBILE_THRESHOLD)) and trajectory.length >= 25]
    lengths_durations = list(set(lengths_durations))
    lengths_durations = sorted(lengths_durations, key=lambda x: (x[0], -x[1]))

    with open('lengths_and_durations_cache.txt', 'w') as file:
        file.write('\n'.join(str(length_duration[0])+','+str(length_duration[1]) for length_duration in lengths_durations))
else:
    with open('lengths_and_durations_cache.txt', 'r') as file:
        lengths_durations = [(int(line.strip().split(',')[0]), float(line.strip().split(',')[1]))  for line in file.readlines()]

reference_architectures = {
    WaveNetTCNFBMModelClassifier: None,
    WaveNetTCNSBMModelClassifier: None,
    WavenetTCNWithLSTMHurstExponentPredicter: {},
}

for index, length_duration in tqdm.tqdm(list(enumerate(lengths_durations))):
    print("Length,Duration:", length_duration)
    length = length_duration[0]
    duration = length_duration[1]
    clear_session()
    for network_class in [WaveNetTCNFBMModelClassifier, WaveNetTCNSBMModelClassifier, WavenetTCNWithLSTMHurstExponentPredicter]:

        if network_class == WavenetTCNWithLSTMHurstExponentPredicter:

            for class_model in ALL_SUB_MODELS:
                already_trained_networks = network_class.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters(class_model.STRING_LABEL))
                available_networks = [network for network in already_trained_networks if network.trajectory_length == length and (network.trajectory_time * 0.85 <= duration <= network.trajectory_time * 1.15) and network.extra_parameters['model'] == class_model.STRING_LABEL]

                if len(available_networks) == 0:
                    classifier = network_class(length, duration, simulator=CustomDataSimulation, model=class_model.STRING_LABEL)

                    if index == 0:
                        reference_architectures[network_class][class_model.STRING_LABEL] = classifier
                    else:
                        classifier.set_wadnet_tcn_encoder(reference_architectures[network_class][class_model.STRING_LABEL], -3)

                    classifier.enable_early_stopping()
                    classifier.enable_database_persistance()
                    classifier.fit()
                    classifier.save()
                else:
                    if index == 0:
                        if len(available_networks) == 1:
                            classifier = available_networks[0]
                        else:
                            classifier = max(*available_networks, key= lambda net: net.trajectory_time)

                        reference_architectures[network_class][class_model.STRING_LABEL] = classifier
                        classifier.enable_database_persistance()
                        classifier.load_as_file()

        else:
            already_trained_networks = network_class.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters())
            available_networks = [network for network in already_trained_networks if network.trajectory_length == length and (network.trajectory_time * 0.85 <= duration <= network.trajectory_time * 1.15)]

            if len(available_networks) == 0:
                classifier = network_class(length, duration, simulator=CustomDataSimulation)

                if index == 0:
                    reference_architectures[network_class] = classifier
                else:
                    classifier.set_wadnet_tcn_encoder(reference_architectures[network_class], -4)

                classifier.enable_early_stopping()
                classifier.enable_database_persistance()
                classifier.fit()
                classifier.save()
            else:
                if index == 0:
                    if len(available_networks) == 1:
                        classifier = available_networks[0]
                    else:
                        classifier = max(*available_networks, key= lambda net: net.trajectory_time)

                    reference_architectures[network_class] = classifier
                    classifier.enable_database_persistance()
                    classifier.load_as_file()


DatabaseHandler.disconnect()