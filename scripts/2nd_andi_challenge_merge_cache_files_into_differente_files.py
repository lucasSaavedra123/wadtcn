import glob
import pandas as pd
import os
import tqdm


def transform_cache_file_chuck_files(cache_files, output_directory, dataset_type):
    t_id = 0
    for cache_i, cache_file_path in enumerate(cache_files):
        cache_dataframe = pd.read_csv(cache_file_path)

        for trajectory_id in tqdm.tqdm(cache_dataframe['id'].unique()):
            trajectory_dataframe = cache_dataframe[cache_dataframe['id'] == trajectory_id]
            trajectory_dataframe = trajectory_dataframe.sort_values('t')
            trajectory_dataframe.to_csv(os.path.join(output_directory, f'{t_id}_{dataset_type}.csv'))
            t_id += 1

DEST_DIRECTORY_TRAIN = '2ndAndiTrajectories'
DEST_DIRECTORY_VAL = '2ndAndiTrajectories_val'

os.makedirs(DEST_DIRECTORY_TRAIN, exist_ok=True)
os.makedirs(DEST_DIRECTORY_VAL, exist_ok=True)

transform_cache_file_chuck_files(glob.glob('*train*classification.cache'), DEST_DIRECTORY_TRAIN, 'classifier')
transform_cache_file_chuck_files(glob.glob('*val*classification.cache'), DEST_DIRECTORY_VAL, 'classifier')

transform_cache_file_chuck_files(glob.glob('*train*regression.cache'), DEST_DIRECTORY_TRAIN, 'regression')
transform_cache_file_chuck_files(glob.glob('*val*regression.cache'), DEST_DIRECTORY_VAL, 'regression')
