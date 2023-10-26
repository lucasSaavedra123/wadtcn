import tqdm
import pandas as pd

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from DataSimulation import CustomDataSimulation

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

f1_score_info = {'length': [] , 'duration': [],'mae_score': [], 'model': []}

print("Loading Hurst Exponent Regression Networks...")
regression_trained_networks = list(WavenetTCNWithLSTMHurstExponentPredicter.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True))
regression_trained_networks = sorted(regression_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))
reference_network = regression_trained_networks[0]

reference_classification_trained_networks = [network for network in regression_trained_networks if network.trajectory_length == reference_network.trajectory_length and network.trajectory_time == reference_network.trajectory_time]

for reference_network in tqdm.tqdm(reference_classification_trained_networks):
    reference_network.enable_database_persistance()
    reference_network.load_as_file()

for transfer_learning_network in tqdm.tqdm([n for n in regression_trained_networks if n not in reference_classification_trained_networks]):
    reference_network = [network for network in reference_classification_trained_networks if transfer_learning_network.extra_parameters['model'] == network.extra_parameters['model']][0]
    transfer_learning_network.set_wadnet_tcn_encoder(reference_network, -3)
    transfer_learning_network.enable_database_persistance()
    transfer_learning_network.load_as_file()

DatabaseHandler.disconnect()

for network in tqdm.tqdm(reference_classification_trained_networks):
    f1_score_info['length'].append(network.trajectory_length)
    f1_score_info['duration'].append(network.trajectory_time)
    f1_score_info['mae_score'].append(network.mae_score())
    f1_score_info['model'].append(network.models_involved_in_predictive_model[0].STRING_LABEL)
    pd.DataFrame(f1_score_info).to_csv('custom_classification_mae.csv')
