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
    trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find(basic_query_dict, {})]

    segments = []
    traj = []

    for trajectory_id in trajectories_ids:
        trajectories = Trajectory.objects(id=trajectory_id)
        assert len(trajectories) == 1
        trajectory = trajectories[0]

        duration = {0:0,1:0}

        if 'analysis' not in trajectory.info or 'run_and_turn_segmenter_result' not in trajectory.info['analysis']:
            continue

        sub_trajectories_by_state = trajectory.sub_trajectories_trajectories_from_confinement_states(states=trajectory.info['analysis']['run_and_turn_segmenter_result'])
        for state in sub_trajectories_by_state:
            for sub_trajectory in sub_trajectories_by_state[state]:
                first_position = np.zeros(2)
                first_position[0] = sub_trajectory.get_noisy_x()[0]
                first_position[1] = sub_trajectory.get_noisy_y()[0]

                second_position = np.zeros(2)
                second_position[0] = sub_trajectory.get_noisy_x()[-1]
                second_position[1] = sub_trajectory.get_noisy_y()[-1]

                distance = np.linalg.norm(first_position-second_position)

                segments.append([state,sub_trajectory.duration,distance,trajectory.info['file'],trajectory.info['roi']])

                duration[state] += sub_trajectory.duration

        rate = np.abs(np.diff(trajectory.info['analysis']['run_and_turn_segmenter_result'])!=0).sum()/trajectory.duration
        traj.append([duration[1]/(duration[0]+duration[1]),rate,trajectory.info['file'],trajectory.info['roi']])

    df = pd.DataFrame(segments, columns=["state", "duration", "distance", "file", "roi"])
    df["state"] = np.vectorize({1:'turn', 0:'run'}.get)(df["state"])
    df.groupby(["state", "file", "roi"]).mean().to_csv(f"{dataset}_run_and_turn_segments_time.csv")

    df = pd.DataFrame(traj, columns=["turn_r", 'transition_rate', "file", "roi"])
    df.groupby(["file", "roi"]).mean().to_csv(f"{dataset}_run_and_turn_traj_time.csv")

DatabaseHandler.disconnect()
