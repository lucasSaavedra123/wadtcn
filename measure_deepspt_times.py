"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import pandas as pd
import numpy as np

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory

APPLY_GS_CRITERIA = True

DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

INDIVIDUAL_DATASETS = [
    'Control',
    'CDx',
    'BTX680R',
    'CholesterolPEGKK114',
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
]:
    new_datasets_list.append((combined_dataset, 'BTX680R'))
    new_datasets_list.append((combined_dataset, 'fPEG-Chol'))

for index, dataset in enumerate(new_datasets_list):
    print(dataset,index)

    basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}
    info_ts = list(Trajectory._get_collection().find(basic_query_dict, {'t':1,'info.analysis.deepspt_segmenter_result':1, 'info.file':1,'info.roi':1}))
    segments = []

    for raw_data in info_ts:
        try:
            data = list(zip(raw_data['t'], raw_data['info']['analysis']['deepspt_segmenter_result']))
            initial = data[0][0]
            current_value = data[0][1]
            traj_duration = raw_data['t'][-1] - raw_data['t'][0]

            for i in range(1, len(data)):
                ts, val = data[i]
                if val != current_value:
                    fin = data[i - 1][0]
                    segments.append([current_value, initial, fin, raw_data['info']['file'], raw_data['info']['roi'], traj_duration])
                    initial = ts
                    current_value = val

            segments.append([current_value, initial, data[-1][0], raw_data['info']['file'], raw_data['info']['roi'], traj_duration])
        except KeyError:
            pass
    
    df = pd.DataFrame(segments, columns=["state", "t_0", "t_1", "file", "roi", "traj_duration"])
    df["state"] = np.vectorize({1:'directed', 0:'normal', 2:'confined', 3:'subdifussive'}.get)(df["state"])
    df['duration'] = df['t_1'] - df['t_0']
    df['proportion_duration'] = df['duration']/df['traj_duration']
    df.groupby(["state", "file", "roi"]).mean().to_csv(f"{dataset}_segments_time.csv")

DatabaseHandler.disconnect()
