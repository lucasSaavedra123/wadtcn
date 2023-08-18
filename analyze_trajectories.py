import tqdm

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation
from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME, IMMOBILE_THRESHOLD
from TheoreticalModels import ALL_MODELS
from Trajectory import Trajectory
from collections import Counter

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

all_trajectories = Trajectory.objects()

number_of_trayectories = len(all_trajectories)

filtered_trajectories = [trajectory for trajectory in all_trajectories if not trajectory.is_immobile(IMMOBILE_THRESHOLD)]

number_of_immobile_trajectories = number_of_trayectories - len(filtered_trajectories)

print(f"There are {number_of_trayectories} trajectories and {number_of_immobile_trajectories} are immobile ({100 * round(number_of_immobile_trajectories/number_of_trayectories, 2)}%).")

DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())

network_and_length = {}

for network in already_trained_networks:
    network.enable_database_persistance()
    network.load_as_file()
    network_and_length[network.trajectory_length] = network

DatabaseHandler.disconnect()

predictions = []

for trajectory in tqdm.tqdm(filtered_trajectories):
    if trajectory.length in network_and_length:
        prediction = network_and_length[trajectory.length].predict([trajectory])
        trajectory.info['model_prediction'] = ALL_MODELS[prediction[0]].STRING_LABEL
        predictions.append(trajectory.info['model_prediction'])
        #trajectory.save()

print(Counter(predictions))