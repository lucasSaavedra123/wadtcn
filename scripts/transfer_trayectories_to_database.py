import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory


DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')
raw_trajectories = list(Trajectory._get_collection().find({}, {}))
DatabaseHandler.disconnect()

DatabaseHandler.connect_over_network(None, None, 'localhost', 'anomalous_diffusion_analysis')
Trajectory._get_collection().insert_many(raw_trajectories)
DatabaseHandler.disconnect()
