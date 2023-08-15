import tqdm
import pandas as pd
import numpy as np
from scipy.io import loadmat


from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from DataSimulation import CustomDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from TheoreticalModels import ALL_MODELS
from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')


from scipy.io import loadmat
from Trajectory import Trajectory
mat_data = loadmat('all_tracks_thunder_localizer.mat')
# Orden en la struct [BTX|mAb] [CDx|Control|CDx-Chol]
dataset = []
# Add each label and condition to the dataset
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][0][0]})
dataset.append({'label': 'BTX',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][0][1]})
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][0][2]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][1][0]})
dataset.append({'label': 'mAb',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][1][1]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][1][2]})

lengths = []

for data in dataset:
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])
    for trajectory in trajectories:
        if not trajectory.is_immobile(1.8) and trajectory.length >= 25:
            lengths.append(trajectory.length)

lengths = np.unique(np.sort(np.array(lengths)))

already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())

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
