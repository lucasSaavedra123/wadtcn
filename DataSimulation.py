from random import shuffle
from numpy.random import choice
import numpy as np
import pandas as pd
import os
from andi_datasets.datasets_challenge import challenge_theory_dataset, _get_dic_andi2, _defaults_andi2
from andi_datasets.datasets_phenom import datasets_phenom

from Trajectory import Trajectory

class DataSimulation():
    STRING_LABEL = 'default'

    def simulate_trajectories_by_model(self, number_of_trajectories, trajectory_length, trajectory_time, model_classes):
        trajectories = []

        while len(trajectories) != number_of_trajectories:
            selected_model_class = choice(model_classes)
            new_trajectory = selected_model_class.create_random_instance().simulate_trajectory(trajectory_length, trajectory_time, from_andi=self.andi)
            trajectories.append(new_trajectory)

        shuffle(trajectories)

        return trajectories

    def simulate_trajectories_by_category(self, number_of_trajectories, trajectory_length, trajectory_time, categories):
        trajectories = []

        while len(trajectories) != number_of_trajectories:
            selected_category = categories[choice(list(range(0,len(categories))))]
            selected_model_class = choice(selected_category)
            new_trajectory = selected_model_class.create_random_instance().simulate_trajectory(trajectory_length, trajectory_time, from_andi=self.andi)
            trajectories.append(new_trajectory)

        shuffle(trajectories)

        return trajectories

class CustomDataSimulation(DataSimulation):
    STRING_LABEL = 'custom'

    def __init__(self):
        self.andi = False

class AndiDataSimulation(DataSimulation):
    STRING_LABEL = 'andi'

    def __init__(self):
        self.andi = True

    def simulate_segmentated_trajectories(self, number_of_trajectories, trajectory_length, trajectory_time):
        _, _, _, _, X, Y = challenge_theory_dataset(number_of_trajectories, min_T=trajectory_length, max_T=trajectory_length+1, tasks=3, dimensions=2)
        trajectories = []


        for trajectory_index in range(number_of_trajectories):
            x = X[1][trajectory_index][:trajectory_length]
            y = X[1][trajectory_index][trajectory_length:]

            simulation_result = {
                'x': x,
                'y': y,
                't': np.arange(0,trajectory_length,1)*trajectory_time/trajectory_length,
                'exponent_type': None,
                'exponent': None,
                'info': {
                    'change_point_time': Y[1][trajectory_index][1],
                    'model_first_segment': Y[1][trajectory_index][2],
                    'alpha_first_segment': Y[1][trajectory_index][3],
                    'model_second_segment': Y[1][trajectory_index][4],
                    'alpha_second_segment': Y[1][trajectory_index][5],
                    }
            }

            trajectories.append(Trajectory(
                    simulation_result['x'],
                    simulation_result['y'],
                    t=simulation_result['t'],
                    #noise_x=simulation_result['x_noisy']-simulation_result['x'],
                    #noise_y=simulation_result['y_noisy']-simulation_result['y'],
                    exponent_type=simulation_result['exponent_type'],
                    exponent=simulation_result['exponent'],
                    model_category=self,
                    info=simulation_result['info'],
                    noisy=True
                )
            )

        return trajectories

class Andi2ndDataSimulation(DataSimulation):
    STRING_LABEL = 'andi2'

    def __init__(self):
        self.andi = True

    def simulate_phenomenological_trajectories(self, number_of_trajectories, trajectory_length, trajectory_time, get_from_cache=False, file_label=''):
        FILE_NAME = f't_{file_label}_{trajectory_length}_{number_of_trajectories}.cache'
        if get_from_cache and os.path.exists(FILE_NAME):
            trajectories = []

            dataframe = pd.read_csv(FILE_NAME)

            for unique_id in dataframe['id'].unique():
                t_dataframe = dataframe[dataframe['id'] == unique_id]
                trajectories.append(Trajectory(
                    x=t_dataframe['x'].tolist(),
                    y=t_dataframe['y'].tolist(),
                    noise_x=(t_dataframe['x_noisy'] - t_dataframe['x']).tolist(),
                    noise_y=(t_dataframe['y_noisy'] - t_dataframe['y']).tolist(),
                    info={
                        'alpha_t': t_dataframe['alpha_t'].tolist(),
                        'd_t': t_dataframe['d_t'].tolist(),
                        'state_t': t_dataframe['state_t'].tolist()
                    }
                ))
        else:
            EXPERIMENTS = np.arange(5).repeat(2)
            NUM_FOVS = 2

            # We create a list of dictionaries with the properties of each experiment
            exp_dic = [None]*len(EXPERIMENTS)

            ##### SINGLE STATE #####
            exp_dic[0] = {'Ds': [1, 0.01], 'alphas' : [0.5, 0.01]}
            exp_dic[1] = {'Ds': [0.1, 0.01], 'alphas' : [1.9, 0.01]}

            ##### MULTI STATE #####
            exp_dic[2] = {'Ds': np.array([[1, 0.01], [0.05, 0.01]]),
                        'alphas' : np.array([[1.5, 0.01], [0.5, 0.01]]),
                        'M': np.array([[0.99, 0.01],[0.01, 0.99]])
                        }
            exp_dic[3] = {'Ds': np.array([[1, 0.01], [0.5, 0.01], [0.01, 0.01]]),
                        'alphas' : np.array([[1.5, 0.01], [0.5, 0.01], [0.75, 0.01]]),
                        'M': np.array([[0.98, 0.01, 0.01],[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])
                        }

            ##### IMMOBILE TRAPS #####
            exp_dic[4] = {'Ds': [1, 0.01], 'alphas' : [0.8, 0.01],
                        'Pu': 0.01, 'Pb': 1,
                        'Nt': 300, 'r': 0.6}
            exp_dic[5] = {'Ds': [1, 0.01], 'alphas' : [1.5, 0.01],
                        'Pu': 0.05, 'Pb': 1,
                        'Nt': 150, 'r': 1}

            ##### DIMERIZATION #####
            exp_dic[6] = {'Ds': np.array([[1, 0.01], [1, 0.01]]), 'alphas' : np.array([[1.2, 0.01], [0.8, 0.01]]),
                        'Pu': 0.01, 'Pb': 1,  'N': 100, 'r': 0.6}
            exp_dic[7] = {'Ds': np.array([[1, 0.01], [3, 0.01]]), 'alphas' : np.array([[1.2, 0.01], [0.5, 0.01]]),
                        'Pu': 0.01, 'Pb': 1,  'N': 80, 'r': 1}

            ##### CONFINEMENT #####
            exp_dic[8] = {'Ds': np.array([[1, 0.01], [1, 0.01]]), 'alphas' : np.array([[0.8, 0.01], [0.4, 0.01]]),
                        'Nc': 30, 'trans': 0.1, 'r': 5}
            exp_dic[9] = {'Ds': np.array([[1, 0.01], [0.1, 0.01]]), 'alphas' : np.array([[1, 0.01], [1, 0.01]]),
                        'Nc': 30, 'trans': 0, 'r': 10}

            trajectories = []

            while len(trajectories) < number_of_trajectories:
                idx = np.random.randint(0, len(EXPERIMENTS))
                if idx < 4:
                    continue
                i = EXPERIMENTS[idx]
                fix_exp = exp_dic[idx]

                dic = _get_dic_andi2(i+1)
                dic['T'] = trajectory_length
                dic['N'] = 100

                for key in fix_exp:
                    dic[key] = fix_exp[key]
                
                def include_trajectory(trajectory):
                    return len(np.unique(trajectory.info['state_t'])) > 1

                for _ in range(NUM_FOVS):
                    trajs, labels = datasets_phenom().create_dataset(dics = dic)
                    trajectories += [ti for ti in Trajectory.from_datasets_phenom(trajs, labels) if include_trajectory(ti)] #We want a diverse number of characteristics

            shuffle(trajectories)
            trajectories = trajectories[:number_of_trajectories]

            if get_from_cache:
                data = {
                    'id':[],
                    'x':[],
                    'y':[],
                    'x_noisy':[],
                    'y_noisy':[],
                    'd_t':[],
                    'alpha_t':[],
                    'state_t':[]
                }

                for i, t in enumerate(trajectories):
                    data['id'] += [i] * t.length
                    data['x'] += t.get_x().tolist()
                    data['y'] += t.get_y().tolist()
                    data['x_noisy'] += t.get_noisy_x().tolist()
                    data['y_noisy'] += t.get_noisy_y().tolist()
                    data['d_t'] += list(t.info['d_t'])
                    data['alpha_t'] += list(t.info['alpha_t'])
                    data['state_t'] += list(t.info['state_t'])

                pd.DataFrame(data).to_csv(FILE_NAME, index=False)

        return trajectories[:number_of_trajectories]
