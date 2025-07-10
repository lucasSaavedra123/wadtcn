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
    prediction = [] 

    intervals = []
    end = len(trajectory_dict['x'])
    step = 1000

    for i in range(0, end, step):
        if i + step < end:
            intervals.append([i, i + step])
        else:
            intervals.append([i, end])

    for interval in intervals:
        trajectory = Trajectory(
            x=trajectory_dict['x'],
            y=trajectory_dict['y'],
            noisy=True
        )

        network.trajectory_length = interval[1] - interval[0]
        prediction += network.predict([trajectory])[0].tolist()

    assert len(prediction) == end
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
