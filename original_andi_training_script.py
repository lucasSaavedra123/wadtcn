import tqdm
import pandas as pd

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.OriginalTheoreticalModelClassifier import OriginalTheoreticalModelClassifier

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

lengths = list(range(25,1000,25))
already_trained_networks = OriginalTheoreticalModelClassifier.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=OriginalTheoreticalModelClassifier.selected_hyperparameters())

length_and_f1_score = {
    'length': [],
    'f1': []
}

for length in tqdm.tqdm(lengths):
    networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

    if len(networks_of_length) == 0:
        classifier = OriginalTheoreticalModelClassifier(length, length, simulator=AndiDataSimulation)
        classifier.enable_early_stopping()
        classifier.fit()
        classifier.save()
    else:
        assert len(networks_of_length) == 1
        classifier = networks_of_length[0]
        classifier.enable_database_persistance()
        classifier.load_as_file()
    
    length_and_f1_score['length'].append(length)
    length_and_f1_score['f1'].append(classifier.model_micro_f1_score())

pd.DataFrame(length_and_f1_score).to_csv('result.csv', index=False)

DatabaseHandler.disconnect()
