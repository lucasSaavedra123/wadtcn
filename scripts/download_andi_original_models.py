import tqdm

from DatabaseHandler import DatabaseHandler
from PredictiveModel.OriginalHurstExponentPredicter import OriginalHurstExponentPredicter
from PredictiveModel.OriginalTheoreticalModelClassifier import OriginalTheoreticalModelClassifier
from PredictiveModel.OriginalFBMModelClassifier import OriginalFBMModelClassifier
from PredictiveModel.OriginalSBMModelClassifier import OriginalSBMModelClassifier
from DataSimulation import AndiDataSimulation


DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

def download_networks(class_name, simulator_class):
    networks = list(class_name.objects(simulator_identifier=simulator_class.STRING_LABEL, trained=True))
    networks = sorted(networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

    for index, network in tqdm.tqdm(list(enumerate(networks))):
        network.enable_database_persistance()
        network.load_as_file()
        network.disable_database_persistance()
        network.save_as_file()

print("Downloading Hurst Exponent Regression Networks...")
download_networks(OriginalHurstExponentPredicter, AndiDataSimulation)

print("Downloading Model Classification Networks...")
download_networks(OriginalTheoreticalModelClassifier, AndiDataSimulation)

print("Downloading fBM Sub-Classification Networks...")
download_networks(OriginalFBMModelClassifier, AndiDataSimulation)

print("Downloading SBM Sub-Classification Networks...")
download_networks(OriginalSBMModelClassifier, AndiDataSimulation)

DatabaseHandler.disconnect()
