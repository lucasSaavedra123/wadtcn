import glob
import pandas as pd
import os
import tqdm


DEST_DIRECTORY = '2ndAndiTrajectories'
os.makedirs(DEST_DIRECTORY, exist_ok=True)
trajectory_counter = len(os.listdir(DEST_DIRECTORY))

for cache_file_path in glob.glob('*.cache'):
    cache_dataframe = pd.read_csv(cache_file_path)

    for trajectory_id in tqdm.tqdm(cache_dataframe['id'].unique()):
        trajectory_dataframe = cache_dataframe[cache_dataframe['id'] == trajectory_id]
        trajectory_dataframe = trajectory_dataframe.sort_values('t')
        trajectory_dataframe.to_csv(os.path.join(DEST_DIRECTORY, f'{str(trajectory_counter).zfill(10)}.csv'), index=False)
        trajectory_counter += 1
