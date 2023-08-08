import tqdm
import pandas as pd

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.OriginalFBMModelClassifier import OriginalFBMModelClassifier
from PredictiveModel.OriginalSBMModelClassifier import OriginalSBMModelClassifier
#from PredictiveModel.Origi import WavenetTCNWithLSTMHurstExponentPredicter

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

lengths = list(range(25,1000,25))

for length in tqdm.tqdm(lengths):
    for network_class in [OriginalFBMModelClassifier, OriginalSBMModelClassifier]: #WavenetTCNWithLSTMHurstExponentPredicter]:

        #if network_class == WavenetTCNWithLSTMHurstExponentPredicter:
        #already_trained_networks = network_class.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=WavenetTCNWithLSTMHurstExponentPredicter.selected_hyperparameters())
        #    pass
        #else:
        
        already_trained_networks = network_class.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters())

        networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

        if len(networks_of_length) == 0:
            classifier = network_class(length, length, simulator=AndiDataSimulation)
            classifier.enable_early_stopping()
            classifier.enable_database_persistance()
            classifier.fit()
            classifier.save()
        else:
            assert len(networks_of_length) == 1
            #classifier = networks_of_length[0]
            #classifier.enable_database_persistance()
            #classifier.load_as_file()


DatabaseHandler.disconnect()