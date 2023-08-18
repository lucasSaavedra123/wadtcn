import tqdm
import pandas as pd
import numpy as np

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from DataSimulation import CustomDataSimulation
from PredictiveModel.ImmobilizedTrajectorySegmentator import ImmobilizedTrajectorySegmentator
from PredictiveModel.ObstructedTrajectorySegmentator import ObstructedTrajectorySegmentator
from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME, IMMOBILE_THRESHOLD


DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')
lengths = np.sort(np.unique([int(trajectory.length) for trajectory in Trajectory.objects() if (not trajectory.is_immobile(IMMOBILE_THRESHOLD)) and trajectory.length >= 25]))
DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

length_and_f1_score = {
    'length': [],
    'f1_od': [],
    'f1_id': []
}

for length in tqdm.tqdm(lengths):
    length_and_f1_score[f'length'].append(length)

    for segmentator in [ObstructedTrajectorySegmentator, ImmobilizedTrajectorySegmentator]:
        already_trained_networks = segmentator.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=segmentator.selected_hyperparameters())

        networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

        if len(networks_of_length) == 0:
            classifier = segmentator(length, length * EXPERIMENT_TIME_FRAME_BY_FRAME, simulator=CustomDataSimulation)
            classifier.enable_early_stopping()
            classifier.enable_database_persistance()
            classifier.fit()
            classifier.save()
        else:
            assert len(networks_of_length) == 1
            classifier = networks_of_length[0]
            classifier.enable_database_persistance()
            classifier.load_as_file()

        length_and_f1_score[f'f1_{segmentator.STRING_LABEL}'].append(classifier.model_micro_f1_score())

    pd.DataFrame(length_and_f1_score).to_csv('segmentation_classification_result.csv', index=False)

DatabaseHandler.disconnect()
