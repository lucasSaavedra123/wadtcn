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

alpha_and_f1_score = {
    'alpha': [],
    'f1_wadtcn': [],
    'f1_tcn': [],
    'f1_lstm': [],
}

randi_lengths = [25,65,125,225,425]
randi_classifiers = []

for length in randi_lengths:
    classifier = LSTMTheoreticalModelClassifier(length, length, simulator=AndiDataSimulation)
    classifier.load_as_file()
    randi_classifiers.append(classifier)

predictions_and_ground_truth_with_certain_alpha = {}

for length in tqdm.tqdm(lengths):
    trajectories = AndiDataSimulation().simulate_trajectories_by_model(12500, length, length, ANDI_MODELS)
    trajectories_with_certain_alpha = {}

    for alpha in np.arange(0.1,2,0.1):
        trajectories_with_certain_alpha[alpha] = []
        if alpha not in predictions_and_ground_truth_with_certain_alpha:
            alpha_and_f1_score['alpha'].append(alpha)
            alpha_and_f1_score['f1_wadtcn'].append(None)
            alpha_and_f1_score['f1_tcn'].append(None)
            alpha_and_f1_score['f1_lstm'].append(None)

            predictions_and_ground_truth_with_certain_alpha[alpha] = {
                'ground truth': [],
                'prediction_f1_wadtcn': [],
                'prediction_f1_tcn': [],
                'prediction_f1_lstm': []
            }

        if alpha < 1:
            choice_models = SUB_DIFFUSIVE_MODELS
        elif alpha == 1:
            choice_models = BROWNIAN_MODELS
        else:
            choice = SUP_DIFFUSIVE_MODELS

        for _ in range(5000):
            model = np.random.choice(ANDI_MODELS)
            model_instance = model.create_random_instance()

            if model == FractionalBrownianMotion:
                model.hurst_exponent = alpha/2
            else:
                model.anomalous_exponent = alpha
            
            trajectories_with_certain_alpha[alpha].append(model_instance.simulate_trajectory(length, length, from_andi=True))

        for trajectory in trajectories_with_certain_alpha[alpha]:
            predictions_and_ground_truth_with_certain_alpha[alpha]['ground truth'].append(ANDI_MODELS.index(trajectory.model_category.__class__))

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
            
            for alpha in trajectories_with_certain_alpha:
                predictions_and_ground_truth_with_certain_alpha[alpha]['prediction_'+info[0]] += picked_network.predict(trajectories_with_certain_alpha[alpha]).tolist()

            if length == 25 or length==500:
                picked_network.plot_confusion_matrix(trajectories=trajectories)
        else:
            predictions = LSTMTheoreticalModelClassifier.classify_with_combination(trajectories, randi_classifiers)
            ground_truth = np.argmax(transform_trajectories_to_categorical_vector(randi_classifiers[0], trajectories), axis=-1)

            length_and_f1_score[info[0]].append(f1_score(ground_truth, predictions, average="micro"))

            for alpha in trajectories_with_certain_alpha:
                predictions_and_ground_truth_with_certain_alpha[alpha]['prediction_'+info[0]] += LSTMTheoreticalModelClassifier.classify_with_combination(trajectories_with_certain_alpha[alpha], randi_classifiers).tolist()

            if length == 25 or length==500:
                confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predictions)
                confusion_mat = np.round(confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis], 2)

                labels = [model.STRING_LABEL for model in ANDI_MODELS]

                confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
                sns.set(font_scale=1.5)
                color_map = sns.color_palette(palette="Blues", n_colors=7)
                sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

                # Plot matrix
                plt.title(f'Confusion Matrix (F1={round(f1_score(ground_truth, predictions, average="micro"),2)})')
                plt.rcParams.update({'font.size': 15})
                plt.ylabel("Ground truth", fontsize=15)
                plt.xlabel("Predicted label", fontsize=15)
                #plt.show()
                plt.savefig(f'randi_length_{length}.jpg')
                plt.clf()


    for index, alpha in enumerate(predictions_and_ground_truth_with_certain_alpha):
        alpha_and_f1_score['f1_wadtcn'][index] = f1_score(predictions_and_ground_truth_with_certain_alpha[alpha]['ground truth'], predictions_and_ground_truth_with_certain_alpha[alpha]['prediction_f1_wadtcn'], average="micro")
        alpha_and_f1_score['f1_tcn'][index] = f1_score(predictions_and_ground_truth_with_certain_alpha[alpha]['ground truth'], predictions_and_ground_truth_with_certain_alpha[alpha]['prediction_f1_tcn'], average="micro")
        alpha_and_f1_score['f1_lstm'][index] = f1_score(predictions_and_ground_truth_with_certain_alpha[alpha]['ground truth'], predictions_and_ground_truth_with_certain_alpha[alpha]['prediction_f1_lstm'], average="micro")

    pd.DataFrame(length_and_f1_score).to_csv('length_classification_result.csv', index=False)
    pd.DataFrame(alpha_and_f1_score).to_csv('alpha_classification_result.csv', index=False)

DatabaseHandler.disconnect()
