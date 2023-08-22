import tqdm
import pandas as pd
from itertools import groupby

from CONSTANTS import *
from TheoreticalModels import ANDI_MODELS
from DataSimulation import AndiDataSimulation
from PredictiveModel.LSTMAnomalousExponentPredicter import LSTMAnomalousExponentPredicter
from PredictiveModel.model_utils import transform_trajectories_to_anomalous_exponent
from sklearn.metrics import f1_score, mean_absolute_error

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from TheoreticalModels import ALL_SUB_MODELS

def get_custom_arquitectures_for_inference(length):
    networks = {}

    class_to_string = {
        WaveNetTCNFBMModelClassifier: 'fbm',
        WaveNetTCNSBMModelClassifier: 'sbm',
        WaveNetTCNTheoreticalModelClassifier: 'general'
    }

    for network_class in [WaveNetTCNFBMModelClassifier, WaveNetTCNSBMModelClassifier, WavenetTCNWithLSTMHurstExponentPredicter, WaveNetTCNTheoreticalModelClassifier]:
        if network_class == WavenetTCNWithLSTMHurstExponentPredicter:

            for class_model in ALL_SUB_MODELS:
                already_trained_networks = network_class.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters(class_model.STRING_LABEL))

                networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length and network.extra_parameters['model'] == class_model.STRING_LABEL]

                assert len(networks_of_length) != 0, 'Not trained yet'
                assert len(networks_of_length) == 1
                classifier = networks_of_length[0]
                classifier.enable_database_persistance()
                classifier.load_as_file()
                classifier.disable_database_persistance()

                networks[f'inference_{class_model.STRING_LABEL}'] = classifier

        else:
            already_trained_networks = network_class.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters())

            networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

            assert len(networks_of_length) != 0, 'Not trained yet'
            assert len(networks_of_length) == 1
            classifier = networks_of_length[0]
            classifier.enable_database_persistance()
            classifier.load_as_file()
            classifier.disable_database_persistance()

            networks[f'classification_{class_to_string[network_class]}'] = classifier

    return networks

randi_lengths = [25,65,125,225,325,425,525,725,925]
randi_classifiers = []

length_and_f1_score = {
    'length': [],
    'mae_lstm': [],
    'mae_wadtcn': []
}

for length in tqdm.tqdm(randi_lengths):
    classifier = LSTMAnomalousExponentPredicter(length, length, simulator=AndiDataSimulation)
    classifier.load_as_file()
    randi_classifiers.append(classifier)

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

lengths = list(range(25,1000,25))

length_to_networks = {}

for length in tqdm.tqdm(lengths):
    try:
        length_to_networks[length] = get_custom_arquitectures_for_inference(length)
    except AssertionError as msg:
        if str(msg) == 'Not trained yet':
            break
        else:
            raise msg

for length in tqdm.tqdm(lengths):
    
    try:
        trajectories = AndiDataSimulation().simulate_trajectories_by_model(12500, length, length, ANDI_MODELS)

        length_and_f1_score['length'].append(length)

        for info in zip(
            ('mae_wadtcn', 'mae_lstm'),
            ('custom', LSTMAnomalousExponentPredicter)
        ):
            
            if info[1] == LSTMAnomalousExponentPredicter:            
                predictions = LSTMAnomalousExponentPredicter.classify_with_combination(trajectories, randi_classifiers)
                ground_truth = transform_trajectories_to_anomalous_exponent(classifier, trajectories)
            else:
                predictions = []
                ground_truth = []

                initial_classifications = length_to_networks[length]['classification_general'].predict(trajectories).tolist()
                initial_classifications = [length_to_networks[length]['classification_general'].models_involved_in_predictive_model[index].STRING_LABEL for index in initial_classifications]

                def model_to_number(x):
                    return [model_class.STRING_LABEL for model_class in length_to_networks[length]['classification_general'].models_involved_in_predictive_model].index(x[0])

                grouped_trajectories = {key: [objeto for _, objeto in group] for key, group in groupby(sorted(zip(initial_classifications, trajectories), key=model_to_number), key=lambda x: x[0])}

                for model_classified_string in grouped_trajectories:
                    model_classified_trayectories = grouped_trajectories[model_classified_string]

                    if model_classified_string == 'fbm' or model_classified_string == 'sbm':
                        sub_classifications = length_to_networks[length][f'classification_{model_classified_string}'].predict(model_classified_trayectories).tolist()
                        sub_classifications = [length_to_networks[length][f'classification_{model_classified_string}'].models_involved_in_predictive_model[index].STRING_LABEL for index in sub_classifications]

                        def sub_model_to_number(x):
                            return [model_class.STRING_LABEL for model_class in length_to_networks[length][f'classification_{model_classified_string}'].models_involved_in_predictive_model].index(x[0])

                        sub_grouped_trajectories = {key: [objeto for _, objeto in group] for key, group in groupby(sorted(zip(sub_classifications, model_classified_trayectories), key=sub_model_to_number), key=lambda x: x[0])}

                        for sub_model_classified_string in sub_grouped_trajectories:
                            ground_truth += [trajectory.anomalous_exponent for trajectory in sub_grouped_trajectories[sub_model_classified_string]]
                            predictions += (length_to_networks[length][f'inference_{sub_model_classified_string}'].predict(sub_grouped_trajectories[sub_model_classified_string])[:,0] * 2).tolist()

                    else:
                        ground_truth += [trajectory.anomalous_exponent for trajectory in model_classified_trayectories]
                        predictions += (length_to_networks[length][f'inference_{model_classified_string}'].predict(model_classified_trayectories)[:,0] * 2).tolist()

            length_and_f1_score[info[0]].append(mean_absolute_error(ground_truth, predictions))

        pd.DataFrame(length_and_f1_score).to_csv('length_inference_result.csv', index=False)
    except KeyError:
        break

DatabaseHandler.disconnect()