import tqdm
import glob
import pandas as pd
import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from DataSimulation import CustomDataSimulation


def extract_tuple_from_file_name(file_name):
    model_section = file_name.split('.')[-2]
    model_section_splitted = model_section.split('_')
    
    if len(model_section_splitted) == 3:
        return int(file_name.split('_')[-4]), float(file_name.split('_')[-3]), model_section_splitted[-1]
    elif len(model_section_splitted) == 4:
        return int(file_name.split('_')[-5]), float(file_name.split('_')[-4]), model_section_splitted[-2] + '_' + model_section_splitted[-1]

FROM_DB = False
f1_score_info = {'length': [] , 'duration': [],'mae_score': [], 'model': []}

print("Loading Hurst Exponent Regression Networks...")
if FROM_DB:
    DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

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
else:
    list_of_files = glob.glob('./networks/wavenet_hurst_exponent_*_*_custom_*')
    lengths_and_durations_and_models = [extract_tuple_from_file_name(file_path) for file_path in list_of_files]
    lengths = np.unique(sorted([length_duration_model[0] for length_duration_model in lengths_and_durations_and_models])).tolist()
    lengths = lengths[::5]

    regression_trained_networks = [WavenetTCNWithLSTMHurstExponentPredicter(a_tuple[0], a_tuple[1], model=a_tuple[2], simulator=CustomDataSimulation) for a_tuple in lengths_and_durations_and_models if a_tuple[0] in lengths]
    regression_trained_networks = sorted(regression_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))
    reference_network = regression_trained_networks[0]
    
    reference_classification_trained_networks = [network for network in regression_trained_networks if network.trajectory_length == reference_network.trajectory_length and network.trajectory_time == reference_network.trajectory_time]

    for reference_network in tqdm.tqdm(reference_classification_trained_networks):
        reference_network.load_as_file()

    for transfer_learning_network in tqdm.tqdm([n for n in regression_trained_networks if n not in reference_classification_trained_networks]):
        reference_network = [network for network in reference_classification_trained_networks if transfer_learning_network.extra_parameters['model'] == network.extra_parameters['model']][0]
        transfer_learning_network.set_wadnet_tcn_encoder(reference_network, -3)
        transfer_learning_network.load_as_file()

for network in tqdm.tqdm(regression_trained_networks):
    f1_score_info['length'].append(network.trajectory_length)
    f1_score_info['duration'].append(network.trajectory_time)
    f1_score_info['mae_score'].append(network.mae_score())
    f1_score_info['model'].append(network.models_involved_in_predictive_model[0].STRING_LABEL)
    pd.DataFrame(f1_score_info).to_csv('custom_classification_mae.csv')
