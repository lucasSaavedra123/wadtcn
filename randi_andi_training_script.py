import tqdm
import pandas as pd

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.LSTMTheoreticalModelClassifier import LSTMTheoreticalModelClassifier

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

#lengths = list(range(25,1000,25))
lengths = [25, 65, 125, 225, 425, 825]
already_trained_networks = LSTMTheoreticalModelClassifier.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=LSTMTheoreticalModelClassifier.selected_hyperparameters())

length_and_f1_score = {
    'length': [],
    'f1': []
}

for length in tqdm.tqdm(lengths):
    networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

    if len(networks_of_length) == 0:
        classifier = LSTMTheoreticalModelClassifier(length, length, simulator=AndiDataSimulation)
        classifier.enable_early_stopping()
        classifier.enable_database_persistance()
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
