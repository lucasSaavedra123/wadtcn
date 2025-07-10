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

trajs = [trajectory_result for trajectory_result in 
       Trajectory._get_collection().find(
           {'info.analysis':{'$exists':True},'info.analysis.deepspt_segmenter_result_probs':{'$exists':False}}, {'_id':1, 'x':1, 'y':1, 'info.analysis':1}
        )
]

for traj in trajs:
    traj['_id'] = str(traj['_id'])

DatabaseHandler.disconnect()

with open("trajs_for_analysis.json", "w") as json_file:
    json.dump(trajs,json_file)
