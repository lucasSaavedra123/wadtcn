import pandas as pd
import tqdm

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation


DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

f1_score_info = {'length': [] , 'duration': [],'f1_score': []}

print("Loading Model Classification Networks...")
classification_trained_networks = list(WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()))
classification_trained_networks = sorted(classification_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in tqdm.tqdm(list(enumerate(classification_trained_networks))):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()

DatabaseHandler.disconnect()

for network in classification_trained_networks:
    f1_score_info['length'].append(network.trajectory_length)
    f1_score_info['duration'].append(network.trajectory_time)
    f1_score_info['f1_score'].append(network.micro_f1_score())
    pd.DataFrame(f1_score_info).to_csv('custom_classification_accuracy.csv')
