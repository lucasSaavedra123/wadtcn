import numpy as np
from DatabaseHandler import DatabaseHandler
from PredictiveModel.RunAndTurnSegmentator import RunAndTurnSegmentator
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from DataSimulation import CustomDataSimulation, DeepSPTDataSimulation
from Trajectory import Trajectory
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils
import json

with open("trajs_for_analysis.json", "r") as json_file:
    trajs = json.load(json_file)

network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(1000,1000,simulator=DeepSPTDataSimulation)
network.load_as_file('wavenet_minflux.weights.h5')
#network = RunAndTurnSegmentator(200,200,simulator=CustomDataSimulation)
#network.load_as_file('run_and_turn_minflux.weights.h5')

for trajectory_dict in tqdm.tqdm(trajs):
    trajectory = Trajectory(
        x=trajectory_dict['x'],
        y=trajectory_dict['y'],
        info=trajectory_dict['info'],
        noisy=True
    )

    if 'analysis' not in trajectory.info:
        continue
    network.trajectory_length = trajectory.length
    prediction = network.predict([trajectory])[0]
    trajectory_dict['ínfo']['analysis']['deepspt_segmenter_result_probs'] = [max(probs) for probs in prediction.tolist()]
    prediction = prediction.argmax(axis=-1)
    prediction = utils.delete_short_changes(prediction, umbral=25)
    trajectory_dict['ínfo']['analysis']['deepspt_segmenter_result'] = prediction.tolist()
    trajectory_dict['ínfo']['analysis']['normal-states-deepspt'] = (prediction==0).astype(int).tolist()
    trajectory_dict['ínfo']['analysis']['directed-states-deepspt'] = (prediction==1).astype(int).tolist()
    trajectory_dict['ínfo']['analysis']['confinement-states-deepspt'] = (prediction==2).astype(int).tolist()
    trajectory_dict['ínfo']['analysis']['subdifussive-states-deepspt'] = (prediction==3).astype(int).tolist()

with open("trajs_for_analysis_predicted.json", "w") as json_file:
    json.dump(trajs,json_file)
