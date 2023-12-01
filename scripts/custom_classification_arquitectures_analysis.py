import pandas as pd
import tqdm
import glob

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation


FROM_DB = False

f1_score_info = {'length': [] , 'duration': [],'f1_score': []}

print("Loading Model Classification Networks...")
if FROM_DB:
    DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

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
else:
    list_of_files = glob.glob('./wavenet_tcn_theoretical_model_classifier_*_custom*')
    lengths_and_durations = [(int(file_path.split('_')[-10]), float(file_path.split('_')[-9])) for file_path in list_of_files]
    lengths_and_durations = sorted(lengths_and_durations, key=lambda a_tuple: (a_tuple[0], -a_tuple[1]))
    classification_trained_networks = []

    for index, length_and_duration in tqdm.tqdm(list(enumerate(lengths_and_durations))):
        network = WaveNetTCNTheoreticalModelClassifier(length_and_duration[0], length_and_duration[1], simulator=CustomDataSimulation)

        if index == 0:
            reference_network = network   
        else:
            network.set_wadnet_tcn_encoder(reference_network, -4)

        network.load_as_file()
        classification_trained_networks.append(network)

for network in tqdm.tqdm(classification_trained_networks):
    f1_score_info['length'].append(network.trajectory_length)
    f1_score_info['duration'].append(network.trajectory_time)
    f1_score_info['f1_score'].append(network.micro_f1_score())
    pd.DataFrame(f1_score_info).to_csv('custom_classification_accuracy.csv')
