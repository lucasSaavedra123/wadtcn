import tqdm
import pandas as pd
import numpy as np

from CONSTANTS import *
from TheoreticalModels import ANDI_MODELS
from DataSimulation import AndiDataSimulation
from PredictiveModel.LSTMTheoreticalModelClassifier import LSTMTheoreticalModelClassifier
from PredictiveModel.model_utils import transform_trajectories_to_categorical_vector
from sklearn.metrics import f1_score

lengths = [25,65,125,225,425]
classifiers = []

length_and_f1_score = {
    'length': [],
    'f1': []
}

for length in tqdm.tqdm(lengths):
    classifier = LSTMTheoreticalModelClassifier(length, length, simulator=AndiDataSimulation)
    classifier.load_as_file()
    classifiers.append(classifier)

for length in tqdm.tqdm(range(25,1000,25)):
    length_and_f1_score['length'].append(length)

    trajectories = AndiDataSimulation().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, length, length, ANDI_MODELS)
    
    predictions = LSTMTheoreticalModelClassifier.classify_with_combination(trajectories, classifiers)
    ground_truth = np.argmax(transform_trajectories_to_categorical_vector(classifier, trajectories), axis=-1)

    length_and_f1_score['f1'].append(f1_score(ground_truth, predictions, average="micro"))

    pd.DataFrame(length_and_f1_score).to_csv('result.csv', index=False)
