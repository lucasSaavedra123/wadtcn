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


DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

with open("trajs_for_analysis_predicted.json", "r") as json_file:
    trajs = json.load(json_file)

for trajectory_dict in tqdm.tqdm(trajs):
    trajectories = Trajectory.objects(id=trajectory_dict['_id'])
    assert len(trajectories) == 1
    trajectory = trajectories[0]

    trajectory.info['analysis']['deepspt_segmenter_result_probs'] = trajectory_dict['info']['analysis']['deepspt_segmenter_result_probs']
    trajectory.info['analysis']['deepspt_segmenter_result'] = trajectory_dict['info']['analysis']['deepspt_segmenter_result']
    trajectory.info['analysis']['normal-states-deepspt'] = trajectory_dict['info']['analysis']['normal-states-deepspt']
    trajectory.info['analysis']['directed-states-deepspt'] = trajectory_dict['info']['analysis']['directed-states-deepspt']
    trajectory.info['analysis']['confinement-states-deepspt'] = trajectory_dict['info']['analysis']['confinement-states-deepspt']
    trajectory.info['analysis']['subdifussive-states-deepspt'] = trajectory_dict['info']['analysis']['subdifussive-states-deepspt']
    trajectory.save()

with open("trajs_for_analysis_predicted.json", "w") as json_file:
    json.dump(trajs,json_file)

DatabaseHandler.disconnect()
