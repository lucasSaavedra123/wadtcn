import tqdm
import pandas as pd

from CONSTANTS import *
from TheoreticalModels import ANDI_MODELS
from DataSimulation import AndiDataSimulation
from PredictiveModel.model_utils import transform_trajectories_to_anomalous_exponent
from sklearn.metrics import f1_score, mean_absolute_error

from PredictiveModel.LSTMAnomalousExponentPredicter import LSTMAnomalousExponentPredicter
from PredictiveModel.inference_utils import get_architectures_for_inference,  infer_with_concatenated_networks

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation

randi_lengths = [25,65,125,225,325,425,525,725,925]
randi_classifiers = []

length_and_mae = {
    'length': [],
    'mae_wadtcn': [],
    'mae_original': [],
    'mae_lstm': [],
}

print("Loading RANDI networks...")
for length in tqdm.tqdm(randi_lengths):
    classifier = LSTMAnomalousExponentPredicter(length, length, simulator=AndiDataSimulation)
    classifier.load_as_file()
    randi_classifiers.append(classifier)

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

lengths = list(range(25,1000,25))

length_to_custom_networks = {}
length_to_original_networks = {}

print("Loading networks from MongoDB...")
for length in tqdm.tqdm(lengths):
    try:
        length_to_custom_networks[length] = get_architectures_for_inference(length, AndiDataSimulation, 'wadtcn')
    except AssertionError as msg:
        if str(msg) == 'Not trained yet':
            pass
        else:
            raise msg

    try:
        length_to_original_networks[length] = get_architectures_for_inference(length, AndiDataSimulation, 'original')
    except AssertionError as msg:
        if str(msg) == 'Not trained yet':
            pass
        else:
            raise msg


for length in tqdm.tqdm(lengths):
    trajectories = AndiDataSimulation().simulate_trajectories_by_model(12500, length, length, ANDI_MODELS)

    length_and_mae['length'].append(length)

    for info in zip(
        ('mae_wadtcn', 'mae_lstm', 'mae_original'),
        ('wadtcn', LSTMAnomalousExponentPredicter, 'original')
    ):
        
        if info[1] == LSTMAnomalousExponentPredicter:            
            predictions = LSTMAnomalousExponentPredicter.classify_with_combination(trajectories, randi_classifiers)
            ground_truth = transform_trajectories_to_anomalous_exponent(classifier, trajectories)
        else:
            try:
                dictionary_to_use = length_to_custom_networks[length] if info[1] == 'wadtcn' else length_to_original_networks[length]
                ground_truth, predictions = infer_with_concatenated_networks(dictionary_to_use, trajectories, return_ground_truth=True)
            except KeyError:
                ground_truth = [0]
                predictions = [0]

        length_and_mae[info[0]].append(mean_absolute_error(ground_truth, predictions))

    pd.DataFrame(length_and_mae).to_csv('length_inference_result.csv', index=False)


DatabaseHandler.disconnect()