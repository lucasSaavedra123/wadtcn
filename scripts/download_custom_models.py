import tqdm

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from DataSimulation import CustomDataSimulation


DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

print("Downloading Hurst Exponent Regression Networks...")
regression_trained_networks = list(WavenetTCNWithLSTMHurstExponentPredicter.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True))
regression_trained_networks = sorted(regression_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))
reference_network = regression_trained_networks[0]

reference_regression_trained_networks = [network for network in regression_trained_networks if network.trajectory_length == reference_network.trajectory_length and network.trajectory_time == reference_network.trajectory_time]

for reference_network in tqdm.tqdm(reference_regression_trained_networks):
    reference_network.enable_database_persistance()
    reference_network.load_as_file()
    reference_network.disable_database_persistance()
    reference_network.save_as_file()

for transfer_learning_network in tqdm.tqdm([n for n in regression_trained_networks if n not in reference_regression_trained_networks]):
    reference_network = [network for network in reference_regression_trained_networks if transfer_learning_network.extra_parameters['model'] == network.extra_parameters['model']][0]
    transfer_learning_network.set_wadnet_tcn_encoder(reference_network, -3)
    transfer_learning_network.enable_database_persistance()
    transfer_learning_network.load_as_file()
    transfer_learning_network.disable_database_persistance()
    transfer_learning_network.save_as_file()

print("Downloading Model Classification Networks...")
classification_trained_networks = list(WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()))
classification_trained_networks = sorted(classification_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in tqdm.tqdm(list(enumerate(classification_trained_networks))):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()
    network.disable_database_persistance()
    network.save_as_file()

print("Downloading fBM Sub-Classification Networks...")
sub_classification_trained_networks = list(WaveNetTCNFBMModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNFBMModelClassifier.selected_hyperparameters()))
sub_classification_trained_networks = sorted(sub_classification_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in tqdm.tqdm(list(enumerate(sub_classification_trained_networks))):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()
    network.disable_database_persistance()
    network.save_as_file()

print("Downloading SBM Sub-Classification Networks...")
sub_classification_trained_networks = list(WaveNetTCNSBMModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNSBMModelClassifier.selected_hyperparameters()))
sub_classification_trained_networks = sorted(sub_classification_trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in tqdm.tqdm(list(enumerate(sub_classification_trained_networks))):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()
    network.disable_database_persistance()
    network.save_as_file()

DatabaseHandler.disconnect()
