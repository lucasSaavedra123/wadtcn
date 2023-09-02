from itertools import groupby

from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.OriginalFBMModelClassifier import OriginalFBMModelClassifier
from PredictiveModel.OriginalSBMModelClassifier import OriginalSBMModelClassifier
from PredictiveModel.OriginalHurstExponentPredicter import OriginalHurstExponentPredicter
from PredictiveModel.OriginalTheoreticalModelClassifier import OriginalTheoreticalModelClassifier
from TheoreticalModels import ALL_SUB_MODELS

def get_arquitectures_for_inference(length, simulator, architecture_label):
    networks = {}

    if architecture_label == 'wadtcn':
        class_to_string = {
            WaveNetTCNFBMModelClassifier: 'fbm',
            WaveNetTCNSBMModelClassifier: 'sbm',
            WaveNetTCNTheoreticalModelClassifier: 'general'
        }

        network_classes = [WaveNetTCNFBMModelClassifier, WaveNetTCNSBMModelClassifier, WavenetTCNWithLSTMHurstExponentPredicter, WaveNetTCNTheoreticalModelClassifier]
        network_class_h_predicter = WavenetTCNWithLSTMHurstExponentPredicter
    elif architecture_label == 'original':
        class_to_string = {
            OriginalFBMModelClassifier: 'fbm',
            OriginalSBMModelClassifier: 'sbm',
            OriginalTheoreticalModelClassifier: 'general'
        }

        network_classes = [OriginalFBMModelClassifier, OriginalSBMModelClassifier, OriginalHurstExponentPredicter, OriginalTheoreticalModelClassifier]
        network_class_h_predicter = OriginalHurstExponentPredicter
    else:
        raise ValueError(f"{simulator} is not a valid anomalous diffusion simulator")

    for network_class in network_classes:
        if network_class == network_class_h_predicter:

            for class_model in ALL_SUB_MODELS:
                already_trained_networks = network_class.objects(simulator_identifier=simulator.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters(class_model.STRING_LABEL))

                networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length and network.extra_parameters['model'] == class_model.STRING_LABEL]

                assert len(networks_of_length) != 0, 'Not trained yet'
                assert len(networks_of_length) == 1
                classifier = networks_of_length[0]
                classifier.enable_database_persistance()
                classifier.load_as_file()
                classifier.disable_database_persistance()

                networks[f'inference_{class_model.STRING_LABEL}'] = classifier

        else:
            already_trained_networks = network_class.objects(simulator_identifier=simulator.STRING_LABEL, trained=True, hyperparameters=network_class.selected_hyperparameters())

            networks_of_length = [network for network in already_trained_networks if network.trajectory_length == length]

            assert len(networks_of_length) != 0, 'Not trained yet'
            assert len(networks_of_length) == 1
            classifier = networks_of_length[0]
            classifier.enable_database_persistance()
            classifier.load_as_file()
            classifier.disable_database_persistance()

            networks[f'classification_{class_to_string[network_class]}'] = classifier

    return networks

def infer_with_concatenated_networks(network_dictionary, trajectories, return_ground_truth=False):
    now_predictions, now_ground_truth = [], []

    initial_classifications = network_dictionary['classification_general'].predict(trajectories).tolist()
    initial_classifications = [network_dictionary['classification_general'].models_involved_in_predictive_model[index].STRING_LABEL for index in initial_classifications]

    def model_to_number(x):
        return [model_class.STRING_LABEL for model_class in network_dictionary['classification_general'].models_involved_in_predictive_model].index(x[0])

    grouped_trajectories = {key: [objeto for _, objeto in group] for key, group in groupby(sorted(zip(initial_classifications, trajectories), key=model_to_number), key=lambda x: x[0])}

    for model_classified_string in grouped_trajectories:
        model_classified_trayectories = grouped_trajectories[model_classified_string]

        if model_classified_string == 'fbm' or model_classified_string == 'sbm':
            sub_classifications = network_dictionary[f'classification_{model_classified_string}'].predict(model_classified_trayectories).tolist()
            sub_classifications = [network_dictionary[f'classification_{model_classified_string}'].models_involved_in_predictive_model[index].STRING_LABEL for index in sub_classifications]

            def sub_model_to_number(x):
                return [model_class.STRING_LABEL for model_class in network_dictionary[f'classification_{model_classified_string}'].models_involved_in_predictive_model].index(x[0])

            sub_grouped_trajectories = {key: [objeto for _, objeto in group] for key, group in groupby(sorted(zip(sub_classifications, model_classified_trayectories), key=sub_model_to_number), key=lambda x: x[0])}

            for sub_model_classified_string in sub_grouped_trajectories:
                now_ground_truth += [trajectory.anomalous_exponent for trajectory in sub_grouped_trajectories[sub_model_classified_string]]
                now_predictions += (network_dictionary[f'inference_{sub_model_classified_string}'].predict(sub_grouped_trajectories[sub_model_classified_string])[:,0] * 2).tolist()

        else:
            now_ground_truth += [trajectory.anomalous_exponent for trajectory in model_classified_trayectories]
            now_predictions += (network_dictionary[f'inference_{model_classified_string}'].predict(model_classified_trayectories)[:,0] * 2).tolist()

    if return_ground_truth:
        return now_ground_truth, now_predictions
    else:
        return now_predictions
