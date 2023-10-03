import pandas as pd

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation


f1_score_and_length = {'length': [] ,'f1_score': []}

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

reference_network = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trajectory_length=25, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())
assert len(reference_network) == 1
network_and_length = {25: reference_network[0]}
network_and_length[25].enable_database_persistance()
network_and_length[25].load_as_file()

for network in WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()):
    if network.trajectory_length != 25:
        network.set_wadnet_tcn_encoder(network_and_length[25], -4)
        network.enable_database_persistance()
        network.load_as_file()
        network_and_length[network.trajectory_length] = network

DatabaseHandler.disconnect()

for length in network_and_length:
    f1_score_and_length['length'].append(length)
    f1_score_and_length['f1_score'].append(network_and_length[length].micro_f1_score())
    pd.DataFrame(f1_score_and_length).to_csv('custom_classification_accuracy.csv')
