import glob
import numpy as np
from Trajectory import Trajectory
import pandas as pd
from DataSimulation import Andi2ndDataSimulation


FROM_TRAINING_POOL=False
if FROM_TRAINING_POOL:
    cache_files = glob.glob('*train*.cache')

    for cache_i, cache_file_path in enumerate(cache_files):
        cache_dataframe = pd.read_csv(cache_file_path)
        
        for trajectory_id in cache_dataframe['id'].unique():
            trajectory_dataframe = cache_dataframe[cache_dataframe['id'] == trajectory_id]
            trajectory_dataframe = trajectory_dataframe.sort_values('t')

            trajectory = Trajectory(
                x = trajectory_dataframe['x_noisy'].tolist(),
                y = trajectory_dataframe['y_noisy'].tolist(),
                t = trajectory_dataframe['t'].tolist(),
                info={
                    'd_t': trajectory_dataframe['d_t'].tolist(),
                    'alpha_t': trajectory_dataframe['alpha_t'].tolist(),
                    'state_t': trajectory_dataframe['state_t'].tolist()
                },
                noisy=True
            ).plot_andi_2()
else:
    t = Andi2ndDataSimulation().simulate_phenomenological_trajectories(100, 100, None, get_from_cache=False, ignore_boundary_effects=True, type_of_simulation='challenge_phenom_dataset')
    for t_i in t:
        #t_i.plot_confinement_states(v_th=19, transition_fix_threshold=1)
        t_i.plot_andi_2()
