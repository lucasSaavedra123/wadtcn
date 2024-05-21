import glob
import numpy as np
from Trajectory import Trajectory
import pandas as pd
from DataSimulation import Andi2ndDataSimulation


FROM_TRAINING_POOL=True

if FROM_TRAINING_POOL:
    ALL_PATHS = glob.glob('./2ndAndiTrajectories/*.csv')

    for file_path in ALL_PATHS:
        t_dataframe = pd.read_csv(file_path)
        Trajectory(
            x=t_dataframe['x_noisy'].tolist(),
            y=t_dataframe['y_noisy'].tolist(),
            t=t_dataframe['t'].tolist(),
            info={
                'alpha_t': t_dataframe['alpha_t'].tolist(),
                'd_t': t_dataframe['d_t'].tolist(),
                'state_t': t_dataframe['state_t'].tolist()
            },
            noisy=True
        ).plot_andi_2()
else:
    t = Andi2ndDataSimulation().simulate_phenomenological_trajectories(100, 100, None, get_from_cache=False, ignore_boundary_effects=False)
    for t_i in t:
        t_i.plot_andi_2()
