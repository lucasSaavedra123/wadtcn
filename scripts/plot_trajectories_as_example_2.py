import numpy as np
import matplotlib.pyplot as plt

from TheoreticalModels.TwoStateObstructedDiffusion import TwoStateObstructedDiffusion
from TheoreticalModels.TwoStateImmobilizedDiffusion import TwoStateImmobilizedDiffusion
from TheoreticalModels import Model

"""
trajectory_info = TwoStateObstructedDiffusion.create_random_instance().custom_simulate_rawly(250, 250 * 0.01)

x = trajectory_info['x_noisy']
y = trajectory_info['y_noisy']

_, ax = plt.subplots()

state_to_color = {1:'red', 0:'black'}
states_as_color = np.vectorize(state_to_color.get)(trajectory_info['info']['state'])

for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
    ax.plot([x1, x2], [y1, y2], states_as_color[i], marker='o')


plt.xticks([])
plt.yticks([])
plt.xlabel('X')
plt.ylabel('Y')
ax.set_box_aspect(1)
plt.savefig('od_example.jpg', dpi=500)
"""
trajectory_info = TwoStateImmobilizedDiffusion.create_random_instance().custom_simulate_rawly(250, 250 * 0.01)

x = trajectory_info['x_noisy']
y = trajectory_info['y_noisy']

_, ax = plt.subplots()

state_to_color = {1:'red', 0:'black'}
states_as_color = np.vectorize(state_to_color.get)(trajectory_info['info']['state'])

for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
    ax.plot([x1, x2], [y1, y2], states_as_color[i], marker='o')


plt.xticks([])
plt.yticks([])
plt.xlabel('X')
plt.ylabel('Y')
ax.set_box_aspect(1)
plt.savefig('id_example.jpg', dpi=500)
