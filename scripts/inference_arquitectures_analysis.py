import tqdm
import pandas as pd
import numpy as np

from CONSTANTS import *
from TheoreticalModels import *
from DataSimulation import AndiDataSimulation
from PredictiveModel.model_utils import transform_trajectories_to_anomalous_exponent, plot_predicted_and_ground_truth_histogram, plot_bias
from sklearn.metrics import mean_absolute_error
from PredictiveModel.LSTMAnomalousExponentPredicter import LSTMAnomalousExponentPredicter
from PredictiveModel.inference_utils import get_architectures_for_inference,  infer_with_concatenated_networks

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation


FROM_DB = False

randi_lengths = [25,65,125,225,325,425,525,725,925]
randi_classifiers = []

length_and_mae = {
    'length': [],
    'mae_wadtcn': [],
    'mae_original': [],
    'mae_lstm': [],
}

theoretical_model_and_mae = {
    'mae_wadtcn': {t_model.STRING_LABEL:{'ground_truth':[], 'predictions': []} for t_model in ANDI_MODELS},
    'mae_original': {t_model.STRING_LABEL:{'ground_truth':[], 'predictions': []} for t_model in ANDI_MODELS},
    'mae_lstm': {t_model.STRING_LABEL:{'ground_truth':[], 'predictions': []} for t_model in ANDI_MODELS},
}

all_info = {
    'mae_wadtcn': {'ground_truth':[], 'predictions': []},
    'mae_original': {'ground_truth':[], 'predictions': []},
    'mae_lstm': {'ground_truth':[], 'predictions': []},
}

print("Loading RANDI networks...")
for length in tqdm.tqdm(randi_lengths):
    classifier = LSTMAnomalousExponentPredicter(length, length, simulator=AndiDataSimulation)
    classifier.load_as_file()
    randi_classifiers.append(classifier)

if FROM_DB:
    DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

lengths = list(range(25,1000,25))

length_to_custom_networks = {}
length_to_original_networks = {}

print("Loading networks...")
for length in tqdm.tqdm(lengths):
    try:
        length_to_custom_networks[length] = get_architectures_for_inference(length, AndiDataSimulation, 'wadtcn', from_db=FROM_DB)
    except AssertionError as msg:
        if str(msg) == 'Not trained yet':
            pass
        else:
            raise msg

    try:
        length_to_original_networks[length] = get_architectures_for_inference(length, AndiDataSimulation, 'original', from_db=FROM_DB)
    except AssertionError as msg:
        if str(msg) == 'Not trained yet':
            pass
        else:
            raise msg

#Overall performance across different lengths
for length in tqdm.tqdm(lengths):
    trajectories = AndiDataSimulation().simulate_trajectories_by_model(12500, length, length, ANDI_MODELS)

    length_and_mae['length'].append(length)

    for info in zip(
        ('mae_wadtcn', 'mae_lstm', 'mae_original'),
        ('wadtcn', LSTMAnomalousExponentPredicter, 'original')
    ):
        
        if info[1] == LSTMAnomalousExponentPredicter:            
            predictions = LSTMAnomalousExponentPredicter.classify_with_combination(trajectories, randi_classifiers).tolist()
            ground_truth = transform_trajectories_to_anomalous_exponent(classifier, trajectories)[:,0].tolist()
        else:
            try:
                dictionary_to_use = length_to_custom_networks[length] if info[1] == 'wadtcn' else length_to_original_networks[length]
                ground_truth, predictions = infer_with_concatenated_networks(dictionary_to_use, trajectories, return_ground_truth=True)
            except KeyError:
                ground_truth = [0]
                predictions = [0]

        length_and_mae[info[0]].append(mean_absolute_error(ground_truth, predictions))

        for i in range(len(predictions)):
            theoretical_model_and_mae[info[0]][trajectories[i].model_category.__class__.STRING_LABEL]['predictions'].append(predictions[i])
            theoretical_model_and_mae[info[0]][trajectories[i].model_category.__class__.STRING_LABEL]['ground_truth'].append(ground_truth[i])

    pd.DataFrame(length_and_mae).to_csv('length_inference_result.csv', index=False)

#Overall performance across different models
trajectories_by_length = {length: [] for length in lengths}

for trajectory_id in range(12500):
    selected_length = np.random.choice(lengths)
    trajectories_by_length[selected_length].append(AndiDataSimulation().simulate_trajectories_by_model(1, selected_length, selected_length, ANDI_MODELS)[0])

for length in trajectories_by_length:
    trajectories = trajectories_by_length[length]

    for info in zip(
        ('mae_wadtcn', 'mae_lstm', 'mae_original'),
        ('wadtcn', LSTMAnomalousExponentPredicter, 'original')
    ):

        if info[1] == LSTMAnomalousExponentPredicter:            
            predictions = LSTMAnomalousExponentPredicter.classify_with_combination(trajectories, randi_classifiers).tolist()
            ground_truth = transform_trajectories_to_anomalous_exponent(classifier, trajectories)[:,0].tolist()
        else:
            try:
                dictionary_to_use = length_to_custom_networks[length] if info[1] == 'wadtcn' else length_to_original_networks[length]
                ground_truth, predictions = infer_with_concatenated_networks(dictionary_to_use, trajectories, return_ground_truth=True)
            except KeyError:
                ground_truth = [0]
                predictions = [0]

        for i in range(len(predictions)):
            theoretical_model_and_mae[info[0]][trajectories[i].model_category.__class__.STRING_LABEL]['predictions'].append(predictions[i])
            theoretical_model_and_mae[info[0]][trajectories[i].model_category.__class__.STRING_LABEL]['ground_truth'].append(ground_truth[i])

for andi_model in ANDI_MODELS:
    for arquitecture_name in ('mae_wadtcn', 'mae_lstm', 'mae_original'):
        p = theoretical_model_and_mae[arquitecture_name][andi_model.STRING_LABEL]['predictions']
        g = theoretical_model_and_mae[arquitecture_name][andi_model.STRING_LABEL]['ground_truth']

        theoretical_model_and_mae[arquitecture_name][andi_model.STRING_LABEL] = mean_absolute_error(g, p)

to_save_theoretical_model_and_mae = {
    'model': [],
    'mae_wadtcn': [],
    'mae_original': [],
    'mae_lstm': [],
}

for andi_model in ANDI_MODELS:
    to_save_theoretical_model_and_mae['model'].append(andi_model.STRING_LABEL)
    for arquitecture_name in ('mae_wadtcn', 'mae_lstm', 'mae_original'):
        to_save_theoretical_model_and_mae[arquitecture_name].append(theoretical_model_and_mae[arquitecture_name][andi_model.STRING_LABEL])

pd.DataFrame(to_save_theoretical_model_and_mae).to_csv('model_inference_result.csv', index=False)

#Overall performance across different alphas
trajectories_by_length = {}

for trajectory_id in range(12500):
    model = np.random.choice(ANDI_MODELS)
    model_instance = model.create_random_instance()

    selected_length = np.random.choice(lengths)
    trajectory = model_instance.simulate_trajectory(selected_length, selected_length, from_andi=True)

    if selected_length not in trajectories_by_length:
        trajectories_by_length[selected_length] = []
    
    trajectories_by_length[selected_length].append(trajectory)

for info in zip([
    ('mae_wadtcn', 'mae_lstm', 'mae_original'),
    ('wadtcn', LSTMAnomalousExponentPredicter, 'original')    
]):
    for length in trajectories_by_length:
        trajectories = trajectories_by_length[length]
        if info[1] == LSTMAnomalousExponentPredicter:            
            predictions = LSTMAnomalousExponentPredicter.classify_with_combination(trajectories, randi_classifiers).tolist()
            ground_truth = transform_trajectories_to_anomalous_exponent(classifier, trajectories)[:,0].tolist()
        else:
            try:
                dictionary_to_use = length_to_custom_networks[length] if info[1] == 'wadtcn' else length_to_original_networks[length]
                ground_truth, predictions = infer_with_concatenated_networks(dictionary_to_use, trajectories, return_ground_truth=True)
            except KeyError:
                ground_truth = [0]
                predictions = [0]

        for i in range(len(predictions)):
            all_info[info[0]]['predictions'].append(predictions[i])
            all_info[info[0]]['ground_truth'].append(ground_truth[i])    

for arquitecture_name in all_info:
    plot_bias(np.array(all_info[arquitecture_name]['ground_truth']), np.array(all_info[arquitecture_name]['predictions']), symbol='alpha', a_range=[-1,1], file_name=f"{arquitecture_name}.png")

for arquitecture_name in all_info:
    result = np.histogram2d(all_info[arquitecture_name]['ground_truth'], all_info[arquitecture_name]['predictions'], bins=50, range=[[0,2], [0,2]])
    dataframe = pd.DataFrame(np.flipud(result[0]), result[2][1:51][::-1], columns=result[1][1:51])
    dataframe.to_csv(f"{arquitecture_name}.csv")

if FROM_DB:
    DatabaseHandler.disconnect()
