import tqdm
import pandas as pd

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from DataSimulation import CustomDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier

from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME, IMMOBILE_THRESHOLD

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')
lengths = set(sorted([trajectory.length for trajectory in Trajectory.objects() if not trajectory.is_immobile(IMMOBILE_THRESHOLD) and trajectory.length >= 25]))
DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())

print("Number of Lengths:", len(lengths))

length_and_f1_score = {
    'length': [],
    'f1': []
}

for length in tqdm.tqdm(lengths):
    print(length)
    networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

    if len(networks_of_length) == 0:
        classifier = WaveNetTCNTheoreticalModelClassifier(length, length * EXPERIMENT_TIME_FRAME_BY_FRAME, simulator=CustomDataSimulation)
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

    if length == 25:
        classifier.plot_confusion_matrix()

    pd.DataFrame(length_and_f1_score).to_csv('result.csv', index=False)

DatabaseHandler.disconnect()
