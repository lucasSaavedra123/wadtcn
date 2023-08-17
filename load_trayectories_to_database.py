from scipy.io import loadmat

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

mat_data = loadmat('all_tracks_thunder_localizer.mat')
# Orden en la struct [BTX|mAb] [CDx|Control|CDx-Chol]
dataset = []
# Add each label and condition to the dataset
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][0][0]})
dataset.append({'label': 'BTX',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][0][1]})
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][0][2]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][1][0]})
dataset.append({'label': 'mAb',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][1][1]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][1][2]})

for data in dataset:
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])
    for trajectory in trajectories:
        trajectories.save()

DatabaseHandler.disconnect()
