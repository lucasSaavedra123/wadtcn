import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from DataSimulation import CustomDataSimulation
from CONSTANTS import IMMOBILE_THRESHOLD
from Trajectory import Trajectory

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

print("Loading trajectories...")
all_trajectories = Trajectory.objects()
filtered_trajectories = [trajectory for trajectory in all_trajectories if not trajectory.is_immobile(IMMOBILE_THRESHOLD) and trajectory.length >= 25 and trajectory.info['prediction']['classified_model'] not in ['od', 'id']]

print("Loading Hurst Exponent Regression Networks...")
reference_classification_trained_networks = [network for network in WavenetTCNWithLSTMHurstExponentPredicter.objects() if network.trajectory_length == reference_network.trajectory_length and network.trajectory_time == reference_network.trajectory_time]
transfer_learning_classification_trained_networks = [network for network in WavenetTCNWithLSTMHurstExponentPredicter.objects() if network.trajectory_length != reference_network.trajectory_length and network.trajectory_time != reference_network.trajectory_time]

for reference_network in tqdm.tqdm(reference_classification_trained_networks):
    reference_network.enable_database_persistance()
    reference_network.load_as_file()

for transfer_learning_network in tqdm.tqdm(transfer_learning_classification_trained_networks):
    reference_network = [network for network in reference_classification_trained_networks if transfer_learning_network.extra_parameters['model'] == network.extra_parameters['model']][0]
    transfer_learning_network.set_wadnet_tcn_encoder(reference_network, -3)
    transfer_learning_network.enable_database_persistance()
    transfer_learning_network.load_as_file()

regression_trained_networks = reference_classification_trained_networks + transfer_learning_classification_trained_networks

for trajectory in tqdm.tqdm(filtered_trajectories):
    if 'sub_classified_model' in trajectory.info['prediction']:
        string_filter = trajectory.info['prediction']['sub_classified_model']
    else:
        string_filter = trajectory.info['prediction']['classified_model']

    available_networks = [network for network in regression_trained_networks if network.trajectory_length == trajectory.length and (network.trajectory_time * 0.85 <= trajectory.duration <= network.trajectory_time * 1.15) and network.extra_parameters['model'] == string_filter]

    if len(available_networks) == 0:
        continue
    elif len(available_networks) == 1:
        network = available_networks[0]
    else:
        network_to_select_index = np.argmin(np.abs(np.array([network.trajectory_time for network in available_networks]) - trajectory.duration))
        network = available_networks[network_to_select_index]

    trajectory.info['prediction']['hurst_exponent'] =  network.predict([trajectory]).tolist()[0]
    trajectory.info['prediction']['hurst_exponent_mae'] = network.history_training_info['val_mae'][-1]

    trajectory.save()

DatabaseHandler.disconnect()
