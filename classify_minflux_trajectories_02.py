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

    """
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
    
    plt.xlim([np.mean(x)-0.50,np.mean(x)+0.50])
    plt.ylim([np.mean(y)-0.50,np.mean(y)+0.50])

    plt.savefig(f'DeepSPT examples/{str(traj_i).zfill(9)}.svg')
    plt.savefig(f'DeepSPT examples/{str(traj_i).zfill(9)}.jpg', dpi=50)
    plt.clf()
    """

with open("trajs_for_analysis_predicted.json", "w") as json_file:
    json.dump(trajs,json_file)

exit()
input("Please Turn On Docker...")

DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

for trajectory_id in tqdm.tqdm(trajs):
    trajectories = Trajectory.objects(id=trajectory_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    if 'analysis' not in trajectory.info or 'deepspt_segmenter_result_probs' in trajectory.info['analysis']:
        continue
    network.trajectory_length = trajectory.length
    try:
        prediction = network.predict([trajectory])[0]
    except:
        continue

    trajectory.info['analysis']['deepspt_segmenter_result_probs'] = [max(probs) for probs in prediction.tolist()]
    prediction = prediction.argmax(axis=-1)
    prediction = utils.delete_short_changes(prediction, umbral=25)
    trajectory.info['analysis']['deepspt_segmenter_result'] = prediction.tolist()
    trajectory.info['analysis']['normal-states-deepspt'] = (prediction==0).astype(int).tolist()
    trajectory.info['analysis']['directed-states-deepspt'] = (prediction==1).astype(int).tolist()
    trajectory.info['analysis']['confinement-states-deepspt'] = (prediction==2).astype(int).tolist()
    trajectory.info['analysis']['subdifussive-states-deepspt'] = (prediction==3).astype(int).tolist()
    trajectory.save()
    """
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
    
    plt.xlim([np.mean(x)-0.50,np.mean(x)+0.50])
    plt.ylim([np.mean(y)-0.50,np.mean(y)+0.50])

    plt.savefig(f'DeepSPT examples/{str(traj_i).zfill(9)}.svg')
    plt.savefig(f'DeepSPT examples/{str(traj_i).zfill(9)}.jpg', dpi=50)
    plt.clf()
    """
