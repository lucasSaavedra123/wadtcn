import tqdm
import pandas as pd
import numpy as np

from CONSTANTS import *
from TheoreticalModels import ANDI_MODELS
from DataSimulation import AndiDataSimulation
from PredictiveModel.LSTMAnomalousExponentPredicter import LSTMAnomalousExponentPredicter
from PredictiveModel.model_utils import transform_trajectories_to_anomalous_exponent
from sklearn.metrics import f1_score, mean_absolute_error

lengths = [25,65,125,225,325,425,525,725,925]
classifiers = []

length_and_f1_score = {
    'length': [],
    'mae': []
}

for length in tqdm.tqdm(lengths):
    classifier = LSTMAnomalousExponentPredicter(length, length, simulator=AndiDataSimulation)
    classifier.load_as_file()
    classifiers.append(classifier)

for length in tqdm.tqdm(range(25,1000,25)):
    length_and_f1_score['length'].append(length)

    trajectories = AndiDataSimulation().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, length, length, ANDI_MODELS)
    
    predictions = LSTMAnomalousExponentPredicter.classify_with_combination(trajectories, classifiers)
    ground_truth = transform_trajectories_to_anomalous_exponent(classifier, trajectories)

    length_and_f1_score['mae'].append(mean_absolute_error(ground_truth, predictions))

    pd.DataFrame(length_and_f1_score).to_csv('result.csv', index=False)
