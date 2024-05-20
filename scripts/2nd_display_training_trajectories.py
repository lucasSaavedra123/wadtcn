import glob
import numpy as np
from Trajectory import Trajectory
import pandas as pd

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
