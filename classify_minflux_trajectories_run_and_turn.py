import numpy as np
from DatabaseHandler import DatabaseHandler
from PredictiveModel.RunAndTurnSegmentator import RunAndTurnSegmentator
from DataSimulation import CustomDataSimulation
from Trajectory import Trajectory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils

DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

network = RunAndTurnSegmentator(1000,1000,simulator=CustomDataSimulation)
network.load_as_file('run_and_turn_minflux.weights.h5')


for trajectory in Trajectory.objects():
    if 'analysis' not in trajectory.info:
        continue
    print(trajectory)
    network.trajectory_length = trajectory.length
    prediction = network.predict([trajectory])[0]

    prediction = prediction.argmax(axis=-1)
    prediction = utils.delete_short_changes(prediction, umbral=5)

    trajectory.info['analysis']['run_and_turn_segmenter_result'] = prediction.tolist()
    trajectory.save()
    """
    x = trajectory.get_noisy_x().tolist()
    y = trajectory.get_noisy_y().tolist()

    state_to_color = {1:'red', 0:'black'}
    state_to_label = {1:'turn', 0:'run'}
    states_as_color = np.vectorize(state_to_color.get)(prediction)

    for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
        plt.plot([x1, x2], [y1, y2], states_as_color[i], alpha=1)

    patches = []
    for state_index in state_to_label:
        patches.append(mpatches.Patch(color=state_to_color[state_index], label=state_to_label[state_index]))

    plt.legend(handles=patches)
    plt.show()
    """

DatabaseHandler.disconnect()
