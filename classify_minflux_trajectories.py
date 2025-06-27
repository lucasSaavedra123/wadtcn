import numpy as np
from DatabaseHandler import DatabaseHandler
from PredictiveModel.RunAndTurnSegmentator import RunAndTurnSegmentator
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from DataSimulation import CustomDataSimulation, DeepSPTDataSimulation
from Trajectory import Trajectory
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

#network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(1000,1000,simulator=DeepSPTDataSimulation)
#network.load_as_file('wavenet_minflux.weights.h5')
network = RunAndTurnSegmentator(200,200,simulator=CustomDataSimulation)
network.load_as_file('run_and_turn_minflux.weights.h5')


def delete_short_changes(signal, umbral=5):
    signal = np.array(signal)
    result = signal.copy()
    actual = signal[0]
    initial_index = 0

    for i in range(1, len(signal)):
        if signal[i] != actual:
            duracion = i - initial_index
            if duracion <= umbral:
                result[initial_index:i+1] = actual
            else:
                actual = signal[i]
                initial_index = i
    return result

for trajectory in Trajectory.objects():
    if 'analysis' not in trajectory.info:
        continue
    print(trajectory)
    network.trajectory_length = trajectory.length
    prediction = network.predict([trajectory])[0]

    prediction = prediction.argmax(axis=-1)

    if 0 not in prediction:
        continue

    #prediction = delete_short_changes(prediction, umbral=25)
    #trajectory.info['analysis']['deepspt_segmenter_result'] = prediction.tolist()
    trajectory.save()

    x = trajectory.get_noisy_x().tolist()
    y = trajectory.get_noisy_y().tolist()

    #state_to_color = {1:'red', 0:'black', 2:'green', 3:'orange'}
    #state_to_label = {1:'directed', 0:'normal', 2:'confined', 3:'subdifussive'}
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

DatabaseHandler.disconnect()
