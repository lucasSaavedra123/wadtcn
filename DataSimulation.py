from random import shuffle
from numpy.random import choice
import numpy as np
import pandas as pd
import os
from andi_datasets.datasets_challenge import challenge_theory_dataset, _get_dic_andi2, _defaults_andi2
from andi_datasets.datasets_phenom import datasets_phenom, models_phenom
import tqdm
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

    def __generate_dict_for_model(self, model_label, trajectory_length, number_of_trajectories, force_directed=False):
        assert 1 <= model_label <= 5
        """
        1: single state
        2: N-state
        3: immobilization
        4: dimerization
        5: confinement
        """
        custom_dic = {}
        D_possible_values = np.logspace(np.log10(models_phenom().bound_D[0]), np.log10(models_phenom().bound_D[1]), num=1000)
        ALPHA_possible_values = np.linspace(models_phenom().bound_alpha[0], models_phenom().bound_alpha[1], num=1000)

        if model_label in [1,3]:
            D = np.random.choice(D_possible_values)
            ALPHA = models_phenom().bound_alpha[1] if force_directed else np.random.choice(ALPHA_possible_values)
            custom_dic.update(
                {
                    'Ds': [D, D*0.01], # mean and variance for D
                    'alphas': np.array([ALPHA, 0.01])
                }
            )

        if model_label in [2,4,5]:
            fast_D = np.random.choice(D_possible_values)
            slow_D = fast_D*np.random.rand()

            alpha1 = models_phenom().bound_alpha[1] if force_directed else np.random.choice(ALPHA_possible_values)
            alpha2 = alpha1*np.random.rand()

            custom_dic.update({
                'Ds': np.array([[fast_D, fast_D*0.01], [slow_D, slow_D*0.01]]),
                'alphas': np.array([[alpha1, 0.01], [alpha2, 0.01]])
            })
        """
        # Particle/trap radius and ninding and unbinding probs for dimerization and immobilization
        if model_label in [3,4]:
            custom_dic.update({'Pu': np.random.uniform(0.01,0.05),                           # Unbinding probability
                        'Pb': np.random.uniform(0.75,1.00)})                             # Binding probabilitiy

        if model_label == 1:
            custom_dic.update({'model': datasets_phenom().avail_models_name[0],
                        'dim': 2})

        if model_label == 2:
            p_1 = np.random.uniform(0.50,1.00)
            p_2 = np.random.uniform(0.50,1.00)
            custom_dic.update({'model': datasets_phenom().avail_models_name[1],
                        'M': np.array([[p_1, 1-p_1],            # Transition Matrix
                                    [1-p_2, p_2]]),
                        'return_state_num': True              # To get the state numeration back, , hence labels.shape = TxNx4
                    })
        
        if model_label == 3:
            custom_dic.update({'model': datasets_phenom().avail_models_name[2],
                        'Nt': 300,            # Number of traps (density = 1 currently)
                        'r': 0.4}             # Size of trap
                    )
        if model_label == 4:
            custom_dic.update({'model': datasets_phenom().avail_models_name[3],
                        'r': 0.6,                 # Size of particles
                        'return_state_num': True  # To get the state numeration back, hence labels.shape = TxNx4
                    })

        if model_label == 5:
            custom_dic.update({'model': datasets_phenom().avail_models_name[4],
                        'r': 5,
                        'Nc': 30,
                        'trans': 0.1})
        """
        dic = _get_dic_andi2(model_label)
        dic['T'] = trajectory_length
        dic['N'] = number_of_trajectories

        for key in custom_dic:
            dic[key] = custom_dic[key]

        return dic

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
            NUM_FOVS = 1

            parameter_simulation_setup = [
                {'model': 1, 'force_directed': True},
                {'model': 1, 'force_directed': False},
                {'model': 2, 'force_directed': True},
                {'model': 2, 'force_directed': False},
                {'model': 3, 'force_directed': True},
                {'model': 3, 'force_directed': False},
                {'model': 4, 'force_directed': True},
                {'model': 4, 'force_directed': False},
            ]

            trajectories = []
            with tqdm.tqdm(total=number_of_trajectories) as pbar:
                while len(trajectories) < number_of_trajectories:
                    simulation_setup = np.random.choice(parameter_simulation_setup)
                    retry = True
                    while retry:
                        dic = self.__generate_dict_for_model(simulation_setup['model']+1, trajectory_length, 10, force_directed=simulation_setup['force_directed'])

                        def include_trajectory(trajectory):
                            return len(np.unique(trajectory.info['d_t'])) > 1

                        for _ in range(NUM_FOVS):
                            trajs, labels = datasets_phenom().create_dataset(dics = dic)
                            new_trajectories = [ti for ti in Trajectory.from_datasets_phenom(trajs, labels) if include_trajectory(ti)][:1]
                            
                            if len(new_trajectories) > 0:
                                retry = False
                                pbar.update(len(new_trajectories))
                                trajectories += new_trajectories #We want a diverse number of characteristics

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
