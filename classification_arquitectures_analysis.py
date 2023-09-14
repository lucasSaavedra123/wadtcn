import tqdm
import pandas as pd
import numpy as np
from PredictiveModel.model_utils import transform_trajectories_to_categorical_vector
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.utils import to_categorical

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from TheoreticalModels import ANDI_MODELS, SUB_DIFFUSIVE_MODELS, SUP_DIFFUSIVE_MODELS,BROWNIAN_MODELS
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotion
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.LSTMTheoreticalModelClassifier import LSTMTheoreticalModelClassifier
from PredictiveModel.OriginalTheoreticalModelClassifier import OriginalTheoreticalModelClassifier
from PredictiveModel.model_utils import transform_trajectories_to_categorical_vector

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

lengths = list(range(25,1000,25))
length_and_f1_score = {
    'length': [],
    'f1_wadtcn': [],
    'f1_tcn': [],
    'f1_lstm': [],
}

alphas = np.arange(0.05,1.95, 0.01)
alpha_and_f1_score = {
    'alpha': [],
    'f1_wadtcn': [],
    'f1_tcn': [],
    'f1_lstm': [],
}

alpha_and_predictions = {}

randi_lengths = [25,65,125,225,425]
randi_classifiers = []

for length in randi_lengths:
    classifier = LSTMTheoreticalModelClassifier(length, length, simulator=AndiDataSimulation)
    classifier.load_as_file()
    randi_classifiers.append(classifier)

"""
for length in tqdm.tqdm(lengths):
    trajectories = AndiDataSimulation().simulate_trajectories_by_model(12500, length, length, ANDI_MODELS)
    length_and_f1_score['length'].append(length)

    for info in zip(
        ('f1_wadtcn', 'f1_tcn', 'f1_lstm'),
        (WaveNetTCNTheoreticalModelClassifier, OriginalTheoreticalModelClassifier, LSTMTheoreticalModelClassifier)
    ):
        if info[1] != LSTMTheoreticalModelClassifier:
            architecture = info[1]        

            already_trained_networks = architecture.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=architecture.selected_hyperparameters())
            picked_network = [network for network in already_trained_networks if network.trajectory_length == length][0]

            picked_network.enable_database_persistance()
            picked_network.load_as_file()

            length_and_f1_score[info[0]].append(picked_network.micro_f1_score(trajectories=trajectories))
        else:
            predictions = LSTMTheoreticalModelClassifier.classify_with_combination(trajectories, randi_classifiers)
            ground_truth = np.argmax(transform_trajectories_to_categorical_vector(randi_classifiers[0], trajectories), axis=-1)
            length_and_f1_score[info[0]].append(f1_score(ground_truth, predictions, average="micro"))

    pd.DataFrame(length_and_f1_score).to_csv('length_classification_result.csv', index=False)
"""
trajectories_and_alpha_by_length = {}

for trajectory_id in range(12500):
    alpha = np.random.choice(alphas)

    if alpha < 0.95:
        choice_models = SUB_DIFFUSIVE_MODELS
    elif 0.95 <= alpha < 1.05:
        choice_models = BROWNIAN_MODELS
    else:
        choice_models = SUP_DIFFUSIVE_MODELS

    model = np.random.choice(choice_models)
    model_instance = model.create_random_instance()

    if model == FractionalBrownianMotion:
        model_instance.hurst_exponent = alpha/2
    else:
        model_instance.anomalous_exponent = alpha

    selected_length = np.random.choice(lengths)
    trajectory = model_instance.simulate_trajectory(selected_length, selected_length, from_andi=True)

    if selected_length not in trajectories_and_alpha_by_length:
        trajectories_and_alpha_by_length[selected_length] = []
    
    trajectories_and_alpha_by_length[selected_length].append((trajectory, alpha))

for trajectory_info in trajectories_and_alpha_by_length.values():

    trajectories = [element[0] for element in trajectory_info]
    alphas = [element[1] for element in trajectory_info]

    for info in zip(
        ('f1_wadtcn', 'f1_tcn', 'f1_lstm'),
        (WaveNetTCNTheoreticalModelClassifier, OriginalTheoreticalModelClassifier, LSTMTheoreticalModelClassifier)
    ):
        if info[1] != LSTMTheoreticalModelClassifier:
            architecture = info[1]        

            already_trained_networks = architecture.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=architecture.selected_hyperparameters())
            picked_network = [network for network in already_trained_networks if network.trajectory_length == length][0]

            picked_network.enable_database_persistance()
            picked_network.load_as_file()

            predictions = picked_network.predict(trajectories)
            ground_truth = np.argmax(transform_trajectories_to_categorical_vector(randi_classifiers[0], trajectories), axis=-1)

        else:
            predictions = LSTMTheoreticalModelClassifier.classify_with_combination(trajectories, randi_classifiers)
            ground_truth = np.argmax(transform_trajectories_to_categorical_vector(randi_classifiers[0], trajectories), axis=-1)


        if info[0] not in alpha_and_predictions:
            alpha_and_predictions[info[0]] = {alpha:{'ground_truth':[], 'prediction:':[]} for alpha in alphas}

        for trajectory_index, alpha in enumerate(alphas):
            alpha_and_predictions[info[0]][alpha]['ground_truth'].append(ground_truth[trajectory_index])
            alpha_and_predictions[info[0]][alpha]['prediction'].append(predictions[trajectory_index])

for alpha in alphas:
    alpha_and_f1_score['alpha'].append(alpha)

    for arquitecture_name in ('f1_wadtcn', 'f1_tcn', 'f1_lstm'):
        alpha_and_f1_score[architecture].append(
            f1_score(
                alpha_and_predictions[arquitecture_name][alpha]['ground_truth'],
                alpha_and_predictions[arquitecture_name][alpha]['prediction']
            )
        )

pd.DataFrame(alpha_and_f1_score).to_csv('alpha_classification_result.csv', index=False)
DatabaseHandler.disconnect()
