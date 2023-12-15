import glob

import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.ImmobilizedTrajectorySegmentator import ImmobilizedTrajectorySegmentator
from DataSimulation import CustomDataSimulation


FROM_DB = False

if FROM_DB:
    DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

reference_lengths = [25,50,100]

if FROM_DB:
    trained_networks = list(ImmobilizedTrajectorySegmentator.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=ImmobilizedTrajectorySegmentator.selected_hyperparameters()))
    trained_networks = sorted(trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))
    reference_network = trained_networks[0]
    reference_network.enable_database_persistance()
    reference_network.load_as_file()
else:
    list_of_files = glob.glob('./networks/immobilized_trajectory_segmentator_*_custom*')
    lengths_and_durations = [(int(file_path.split('_')[-4]), float(file_path.split('_')[-3])) for file_path in list_of_files]
    lengths_and_durations = sorted(lengths_and_durations, key=lambda a_tuple: (a_tuple[0], -a_tuple[1]))
    trained_networks = [ImmobilizedTrajectorySegmentator(length_and_duration[0], length_and_duration[1], simulator=CustomDataSimulation) for length_and_duration in lengths_and_durations]
    reference_network = trained_networks[0]
    reference_network.load_as_file()

for length in reference_lengths:
    available_networks = [network for network in trained_networks if network.trajectory_length == length]

    if len(available_networks) == 0:
        continue
    else:
        if len(available_networks) == 1:
            network = available_networks[0]
        else:
            network_to_select_index = np.argmin(np.array([network.trajectory_time for network in available_networks]))
            network = available_networks[network_to_select_index]
        
        if FROM_DB:
            network.enable_database_persistance()

        if network != reference_network:
            network.set_wadnet_tcn_encoder(reference_network, -4)
            network.load_as_file()

        network.plot_confusion_matrix()

if FROM_DB:
    DatabaseHandler.disconnect()
