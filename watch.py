import numpy as np
from DataSimulation import DeepSPTDataSimulation
from Trajectory import Trajectory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


while True:
    trajectory = DeepSPTDataSimulation().simulate_segmentated_trajectories(1,1_000,None, enable_parallelism=False)[0]

    x = trajectory.get_noisy_x().tolist()
    y = trajectory.get_noisy_y().tolist()

    prediction = trajectory.info['state_t']

    state_to_color = {1:'red', 0:'black', 2:'green', 3:'orange'}
    state_to_label = {1:'directed', 0:'normal', 2:'confined', 3:'subdifussive'}
    states_as_color = np.vectorize(state_to_color.get)(prediction)

    for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
        plt.plot([x1, x2], [y1, y2], states_as_color[i], alpha=1)

    patches = []
    for state_index in state_to_label:
        patches.append(mpatches.Patch(color=state_to_color[state_index], label=state_to_label[state_index]))

    plt.legend(handles=patches)
    plt.show()
