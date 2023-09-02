import tqdm
import pandas as pd
from tensorflow.keras.backend import clear_session

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.OriginalFBMModelClassifier import OriginalFBMModelClassifier
from PredictiveModel.OriginalSBMModelClassifier import OriginalSBMModelClassifier
from PredictiveModel.OriginalHurstExponentPredicter import OriginalHurstExponentPredicter
from TheoreticalModels import ALL_SUB_MODELS

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

lengths = list(range(25,1000,25))

for length in tqdm.tqdm(lengths):
    clear_session()
    for network_class in [OriginalFBMModelClassifier, OriginalSBMModelClassifier, OriginalHurstExponentPredicter]:

        if network_class == OriginalHurstExponentPredicter:

            for class_model in ALL_SUB_MODELS:
                already_trained_networks = network_class.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters(class_model.STRING_LABEL))

                networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length and network.extra_parameters['model'] == class_model.STRING_LABEL]

                if len(networks_of_length) == 0:
                    classifier = network_class(length, length, simulator=AndiDataSimulation, model=class_model.STRING_LABEL)
                    classifier.enable_early_stopping()
                    classifier.enable_database_persistance()
                    classifier.fit()
                    classifier.save()
                else:
                    assert len(networks_of_length) == 1
                    classifier = networks_of_length[0]
                    classifier.enable_database_persistance()
                    classifier.load_as_file()

        else:
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
                classifier = networks_of_length[0]
                classifier.enable_database_persistance()
                classifier.load_as_file()


DatabaseHandler.disconnect()