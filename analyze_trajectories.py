import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation
from CONSTANTS import IMMOBILE_THRESHOLD
from TheoreticalModels import ALL_MODELS
from Trajectory import Trajectory

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

all_trajectories = Trajectory.objects()

trained_networks = list(WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()))
trained_networks = sorted(trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in enumerate(trained_networks):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]
        filtered_trajectories = [trajectory for trajectory in filtered_trajectories if not trajectory.is_immobile(IMMOBILE_THRESHOLD) and trajectory.length >= 25]

        predictions = []

        for trajectory in tqdm.tqdm(filtered_trajectories):
            available_networks = [network for network in trained_networks if network.trajectory_length == trajectory.length and (network.trajectory_time * 0.85 <= trajectory.duration <= network.trajectory_time * 1.15)]

            if len(available_networks) == 0:
                continue
            elif len(available_networks) == 1:
                network = available_networks[0]
            else:
                network_to_select_index = np.argmin(np.abs(np.array([network.trajectory_time for network in available_networks]) - trajectory.duration))
                network = available_networks[network_to_select_index]

            trajectory.info['prediction'] = {
                'classified_model': [ ALL_MODELS[i].STRING_LABEL for i in network.predict([trajectory]).tolist()][0],
                'model_classification_accuracy': network.history_training_info['val_categorical_accuracy'][-1],
            }

            trajectory.save()

DatabaseHandler.disconnect()
