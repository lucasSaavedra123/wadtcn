import tqdm
import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter import WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter
from DataSimulation import CustomDataSimulation
from CONSTANTS import IMMOBILE_THRESHOLD
from Trajectory import Trajectory

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

print("Loading trajectories...")
filtered_trajectories = [trajectory for trajectory in Trajectory.objects() if not trajectory.is_immobile(IMMOBILE_THRESHOLD) and trajectory.length >= 25 and trajectory.info['prediction']['classified_model'] == 'fbm']

print("Loading Hurst Exponent Regression Networks...")
regression_trained_networks = list(WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True))
regression_trained_networks = sorted(regression_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in tqdm.tqdm(list(enumerate(regression_trained_networks))):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -3)

    network.enable_database_persistance()
    network.load_as_file()

for trajectory in tqdm.tqdm(filtered_trajectories):
    available_networks = [network for network in regression_trained_networks if network.trajectory_length == trajectory.length and (network.trajectory_time * 0.85 <= round(trajectory.duration, 2) <= network.trajectory_time * 1.15)]

    if len(available_networks) == 0:
        raise Exception(f'There is no available network for {trajectory.length} and {trajectory.duration}s')
    elif len(available_networks) == 1:
        network = available_networks[0]
    else:
        network_to_select_index = np.argmin(np.abs(np.array([network.trajectory_time for network in available_networks]) - trajectory.duration))
        network = available_networks[network_to_select_index]

    trajectory.info['prediction']['diffusion_coefficient'] =  network.predict([trajectory]).tolist()[0]
    trajectory.info['prediction']['diffusion_coefficient_mae'] = network.history_training_info['val_mae'][-1]

    trajectory.save()

DatabaseHandler.disconnect()
