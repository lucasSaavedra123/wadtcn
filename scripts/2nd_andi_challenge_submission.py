#Code reference https://github.com/AnDiChallenge/andi_datasets/blob/master/source_nbs/tutorials/challenge_two_submission.ipynb
import os
import numpy as np
from andi_datasets.datasets_phenom import datasets_phenom
import pandas as pd

public_data_path = 'public_dat/'
path_results = 'res/'
if not os.path.exists(path_results):
    os.makedirs(path_results)

for track in [1,2]:
    # Create the folder of the track if it does not exists
    path_track = path_results + f'track_{track}/'
    if not os.path.exists(path_track):
        os.makedirs(path_track)

    for exp in range(10):
        # Create the folder of the experiment if it does not exits
        path_exp = path_track+f'exp_{exp}/'
        if not os.path.exists(path_exp):
            os.makedirs(path_exp)
        file_name = path_exp + 'ensemble_labels.txt'
        
        with open(file_name, 'a') as f:
            # Save the model (random) and the number of states (2 in this case)
            model_name = np.random.choice(datasets_phenom().avail_models_name, size = 1)[0]
            f.write(f'model: {model_name}; num_state: {2} \n')

            # Create some dummy data for 2 states. This means 2 columns
            # and 5 rows
            data = np.random.rand(5, 2)
            
            data[-1,:] /= data[-1,:].sum()

            # Save the data in the corresponding ensemble file
            np.savetxt(f, data, delimiter = ';')

# Define the number of experiments and number of FOVS
N_EXP = 10 
N_FOVS = 30

# We only to track 2 in this example
track = 2

# The results go in the same folders generated above
path_results = 'res/'
path_track = path_results + f'track_{track}/'

for exp in range(N_EXP):
    
    path_exp = path_track + f'exp_{exp}/'
    
    for fov in range(N_FOVS):
        
        # We read the corresponding csv file from the public data and extract the indices of the trajectories:
        df = pd.read_csv(public_data_path+f'track_2/exp_{exp}/trajs_fov_{fov}.csv')
        traj_idx = df.traj_idx.unique()
        
        submission_file = path_exp + f'fov_{fov}.txt'
        
        with open(submission_file, 'a') as f:
            
            # Loop over each index
            for idx in traj_idx:
                
                # Get the lenght of the trajectory
                length_traj = df[df.traj_idx == traj_idx[0]].shape[0]
                # Assign one changepoints for each traj at 0.25 of its length
                CP = int(length_traj*0.25)
                
                prediction_traj = [idx.astype(int), 
                                   np.random.rand()*10, # K1
                                   np.random.rand(), # alpha1
                                   np.random.randint(4), # state1
                                   CP, # changepoint
                                   np.random.rand()*10, # K2
                                   np.random.rand(), # alpha2
                                   np.random.randint(4), # state2
                                   length_traj # Total length of the trajectory
                                  ]
                
                formatted_numbers = ','.join(map(str, prediction_traj))
                f.write(formatted_numbers + '\n')