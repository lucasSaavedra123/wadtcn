import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation


DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

reference_lengths = [25,50,100]

trained_networks = list(WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()))

reference_network = [network for network in trained_networks if network.trajectory_time==0.44 and network.trajectory_length==25][0]
reference_network.enable_database_persistance()
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
        
        network.set_wadnet_tcn_encoder(reference_network, -4)
        network.enable_database_persistance()
        network.load_as_file()

        print("Probando...")
        network.plot_confusion_matrix()

DatabaseHandler.disconnect()
