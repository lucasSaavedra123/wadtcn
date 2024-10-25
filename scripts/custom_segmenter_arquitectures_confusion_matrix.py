import glob

import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.ImmobilizedTrajectorySegmentator import ImmobilizedTrajectorySegmentator
from DataSimulation import CustomDataSimulation


FROM_DB = True

if FROM_DB:
    DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

reference_lengths = [25,50,96]

trained_networks = list(ImmobilizedTrajectorySegmentator.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=ImmobilizedTrajectorySegmentator.selected_hyperparameters()))
trained_networks = sorted(trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for length in reference_lengths:
    available_networks = [network for network in trained_networks if network.trajectory_length == length]

    if len(available_networks) == 1:
        network = available_networks[0]
    else:
        network_to_select_index = np.argmin(np.array([network.trajectory_time for network in available_networks]))
        network = available_networks[network_to_select_index]
    
    if FROM_DB:
        network.enable_database_persistance()
    network.load_as_file()
    network.plot_confusion_matrix()

if FROM_DB:
    DatabaseHandler.disconnect()
