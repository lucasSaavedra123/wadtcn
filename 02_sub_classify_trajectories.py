import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from DataSimulation import CustomDataSimulation
from CONSTANTS import IMMOBILE_THRESHOLD
from Trajectory import Trajectory

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

print("Loading trajectories...")
filtered_trajectories = [trajectory for trajectory in Trajectory.objects() if not trajectory.is_immobile(IMMOBILE_THRESHOLD) and trajectory.length >= 25 and trajectory.info['prediction']['classified_model'] in ['fbm', 'sbm']]

classification_trained_networks = {}

print("Loading fBM Sub-Classification Networks...")
classification_trained_networks['fbm'] = list(WaveNetTCNFBMModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNFBMModelClassifier.selected_hyperparameters()))
classification_trained_networks['fbm'] = sorted(classification_trained_networks['fbm'], key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in tqdm.tqdm(list(enumerate(classification_trained_networks['fbm']))):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()

print("Loading SBM Sub-Classification Networks...")
classification_trained_networks['sbm'] = list(WaveNetTCNSBMModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNSBMModelClassifier.selected_hyperparameters()))
classification_trained_networks['sbm'] = sorted(classification_trained_networks['sbm'], key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in tqdm.tqdm(list(enumerate(classification_trained_networks['sbm']))):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()


for trajectory in tqdm.tqdm(filtered_trajectories):
    available_networks = [network for network in classification_trained_networks[trajectory.info['prediction']['classified_model']] if network.trajectory_length == trajectory.length and (network.trajectory_time * 0.85 <= trajectory.duration <= network.trajectory_time * 1.15)]

    if len(available_networks) == 0:
        continue
    elif len(available_networks) == 1:
        network = available_networks[0]
    else:
        network_to_select_index = np.argmin(np.abs(np.array([network.trajectory_time for network in available_networks]) - trajectory.duration))
        network = available_networks[network_to_select_index]

    trajectory.info['prediction']['sub_classified_model'] = [ network.models_involved_in_predictive_model[i].STRING_LABEL for i in network.predict([trajectory]).tolist()][0]
    trajectory.info['prediction']['sub_model_classification_accuracy'] = network.history_training_info['val_categorical_accuracy'][-1]

    trajectory.save()

DatabaseHandler.disconnect()
