import tqdm
import pandas as pd

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier

REFERENCE_LENGTH = 500
LENGTHS = list(range(25,1000,25))
f1_scores = []

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())
reference_length_length_classifier = [network for network in already_trained_networks if network.trajectory_length == REFERENCE_LENGTH][0]
reference_length_length_classifier.enable_database_persistance()
reference_length_length_classifier.load_as_file()

for length in tqdm.tqdm(LENGTHS):
    if length == REFERENCE_LENGTH:
        f1_scores.append(reference_length_length_classifier.micro_f1_score())
    else:
        classifier = [network for network in already_trained_networks if network.trajectory_length == length][0]
        classifier.enable_database_persistance()
        classifier.load_as_file()

        classifier.architecture.set_weights(reference_length_length_classifier.architecture.get_weights())
        f1_scores.append(classifier.micro_f1_score())

pd.DataFrame({
    'lengths': LENGTHS,
    'f1_score': f1_scores
}).to_csv(f'length_{REFERENCE_LENGTH}_to_length.csv')

DatabaseHandler.disconnect()
