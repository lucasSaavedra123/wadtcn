import tqdm
import glob
import pandas as pd

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter import WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter
from DataSimulation import CustomDataSimulation


FROM_DB = False
f1_score_info = {'length': [] , 'duration': [],'mae_score': []}

print("Loading Hurst Exponent Regression Networks...")
if FROM_DB:
    DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

    regression_trained_networks = list(WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()))
    regression_trained_networks = sorted(regression_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

    for index, network in tqdm.tqdm(list(enumerate(regression_trained_networks))):
        if index == 0:
            reference_network = network   
        else:
            network.set_wadnet_tcn_encoder(reference_network, -3)

        network.enable_database_persistance()
        network.load_as_file()

    DatabaseHandler.disconnect()
else:
    list_of_files = glob.glob('./networks/wavenet_diffusion_coefficient_*_custom*')
    lengths_and_durations = [(int(file_path.split('_')[-4]), float(file_path.split('_')[-3])) for file_path in list_of_files]
    lengths_and_durations = sorted(lengths_and_durations, key=lambda a_tuple: (a_tuple[0], -a_tuple[1]))
    regression_trained_networks = []

    for index, length_and_duration in tqdm.tqdm(list(enumerate(lengths_and_durations))):
        network = WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter(length_and_duration[0], length_and_duration[1], simulator=CustomDataSimulation)

        if index == 0:
            reference_network = network   
        else:
            network.set_wadnet_tcn_encoder(reference_network, -3)

        network.load_as_file()
        regression_trained_networks.append(network)

for network in tqdm.tqdm(regression_trained_networks):
    f1_score_info['length'].append(network.trajectory_length)
    f1_score_info['duration'].append(network.trajectory_time)
    f1_score_info['mae_score'].append(network.mae_score())
    pd.DataFrame(f1_score_info).to_csv('custom_d_mae.csv')
