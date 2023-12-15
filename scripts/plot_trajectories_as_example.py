import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation
from CONSTANTS import IMMOBILE_THRESHOLD
from TheoreticalModels import ALL_MODELS, Model
from Trajectory import Trajectory


IDS_TO_PLOT = {
    'lw': '6514af77e7868d448ea77f73',
    'ctrw': '6514abbfe7868d448ea727a5',
    'sbm': '6514ac74e7868d448ea73820',
    'fbm': '6514ac46e7868d448ea733c5',
    'attm': '6514ae30e7868d448ea76249',
    'od': '6514ac0fe7868d448ea72e8c',
    'id': '6514b1fbe7868d448ea7c1e3'
}

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

offset = 600

offset_dictionary = {
    6: np.array([0,0]),
    1: np.array([0,1]),
    2: np.array([1,0]),
    3: np.array([1,1]),
    4: np.array([-1,0]),
    5: np.array([0,-1]),
    0: np.array([-1,-1]),
}

_, ax = plt.subplots()

for index, key in enumerate(IDS_TO_PLOT):
    trajectory = Trajectory.objects(id=IDS_TO_PLOT[key])[0]

    x = trajectory.get_noisy_x()
    y = trajectory.get_noisy_y()

    x = (x - np.mean(x)) + (750 * index)# + (offset_dictionary[index][0] * offset)
    y = (y - np.mean(y))# + (offset_dictionary[index][1] * offset)

    ax.plot(x, y, color=Model.Model.MODEL_COLORS[key], linewidth=1.25)

plt.xticks([])
plt.yticks([])
ax.set_aspect(1)
plt.tight_layout()
plt.savefig('example.jpg', dpi=500)


DatabaseHandler.disconnect()
