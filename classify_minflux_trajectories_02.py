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
            x=trajectory_dict['x'][interval[0]:interval[1]],
            y=trajectory_dict['y'][interval[0]:interval[1]],
            noisy=True
        )

        network.trajectory_length = interval[1] - interval[0]
        prediction += network.predict([trajectory])[0].tolist()

    prediction = np.array(prediction)
    assert len(prediction) == end
    trajectory_dict['info']['analysis']['deepspt_segmenter_result_probs'] = [max(probs) for probs in prediction.tolist()]
    prediction = prediction.argmax(axis=-1)
    prediction = utils.delete_short_changes(prediction, umbral=25)
    trajectory_dict['info']['analysis']['deepspt_segmenter_result'] = prediction.tolist()
    trajectory_dict['info']['analysis']['normal-states-deepspt'] = (prediction==0).astype(int).tolist()
    trajectory_dict['info']['analysis']['directed-states-deepspt'] = (prediction==1).astype(int).tolist()
    trajectory_dict['info']['analysis']['confinement-states-deepspt'] = (prediction==2).astype(int).tolist()
    trajectory_dict['info']['analysis']['subdifussive-states-deepspt'] = (prediction==3).astype(int).tolist()
    """
    trajectory = Trajectory(
        x=trajectory_dict['x'],
        y=trajectory_dict['y'],
        noisy=True
    )

    x = trajectory.get_noisy_x().tolist()
    y = trajectory.get_noisy_y().tolist()

    state_to_color = {1:'red', 0:'black', 2:'green', 3:'orange'}
    state_to_label = {1:'Directed', 0:'Normal', 2:'Confined', 3:'Subdifussive'}
    #state_to_color = {1:'red', 0:'black'}
    #state_to_label = {1:'turn', 0:'run'}
    states_as_color = np.vectorize(state_to_color.get)(prediction)

    for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
        plt.plot([x1, x2], [y1, y2], states_as_color[i], alpha=1)

    patches = []
    for state_index in state_to_label:
        patches.append(mpatches.Patch(color=state_to_color[state_index], label=state_to_label[state_index]))

    plt.legend(handles=patches)
    plt.show()
    """

with open("trajs_for_analysis_predicted.json", "w") as json_file:
    json.dump(trajs,json_file)
