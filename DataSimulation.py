import glob
import os

import numpy as np
import pandas as pd
import tqdm
import ray
from andi_datasets.datasets_challenge import challenge_theory_dataset, _get_dic_andi2, challenge_phenom_dataset
from andi_datasets.datasets_phenom import datasets_phenom, models_phenom

from Trajectory import Trajectory


class DataSimulation():
    STRING_LABEL = 'default'

    def simulate_trajectories_by_model(self, number_of_trajectories, trajectory_length, trajectory_time, model_classes):
        trajectories = []

        while len(trajectories) != number_of_trajectories:
            selected_model_class = np.random.choice(model_classes)
            new_trajectory = selected_model_class.create_random_instance().simulate_trajectory(trajectory_length, trajectory_time, from_andi=self.andi)
            trajectories.append(new_trajectory)

        np.random.shuffle(trajectories)

        return trajectories

    def simulate_trajectories_by_category(self, number_of_trajectories, trajectory_length, trajectory_time, categories):
        trajectories = []

        while len(trajectories) != number_of_trajectories:
            selected_category = categories[np.random.choice(list(range(0,len(categories))))]
            selected_model_class = np.random.choice(selected_category)
            new_trajectory = selected_model_class.create_random_instance().simulate_trajectory(trajectory_length, trajectory_time, from_andi=self.andi)
            trajectories.append(new_trajectory)

        np.random.shuffle(trajectories)

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
                    'change_point_time': int(Y[1][trajectory_index][1]),
                    'model_first_segment': Y[1][trajectory_index][2],
                    'alpha_first_segment': Y[1][trajectory_index][3],
                    'model_second_segment': Y[1][trajectory_index][4],
                    'alpha_second_segment': Y[1][trajectory_index][5],
                    }
            }

            alpha_t = np.full(simulation_result['info']['change_point_time'], fill_value=simulation_result['info']['alpha_first_segment']).tolist()
            alpha_t += np.full(trajectory_length-simulation_result['info']['change_point_time'], fill_value=simulation_result['info']['alpha_second_segment']).tolist()

            state_t = np.full(simulation_result['info']['change_point_time'], fill_value=simulation_result['info']['model_first_segment']).tolist()
            state_t += np.full(trajectory_length-simulation_result['info']['change_point_time'], fill_value=simulation_result['info']['model_second_segment']).tolist()

            simulation_result['info']['state_t'] = state_t
            simulation_result['info']['alpha_t'] = alpha_t

            trajectories.append(Trajectory(
                    simulation_result['x'],
                    simulation_result['y'],
                    t=simulation_result['t'],
                    #noise_x=simulation_result['x_noisy']-simulation_result['x'],
                    #noise_y=simulation_result['y_noisy']-simulation_result['y'],
                    exponent_type=simulation_result['exponent_type'],
                    exponent=simulation_result['exponent'],
                    info=simulation_result['info'],
                    noisy=True
                )
            )

        return trajectories

class Andi2ndDataSimulation(DataSimulation):
    STRING_LABEL = 'andi2'

    def __init__(self):
        self.andi = True

    def __generate_dict_for_model(self, model_label, trajectory_length, number_of_trajectories, force_directed=False, ignore_boundary_effects=True, L=128):
        assert 1 <= model_label <= 5
        """
        1: single state
        2: N-state
        3: immobilization
        4: dimerization
        5: confinement
        """
        #Initial settings
        TRAJECTORY_DENSITY = 50/((1.5*128)**2)
        TRAP_DENSITY = 10/((128)**2)
        CONFINEMENTS_DENSITY = 10/((1.5*128)**2)

        MIN_D, MAX_D = models_phenom().bound_D[0], models_phenom().bound_D[1]
        MIN_A, MAX_A = models_phenom().bound_alpha[0], models_phenom().bound_alpha[1]
        custom_dic = {}
        ALPHA_possible_values = np.linspace(MIN_A, MAX_A, num=10000)[1:]
        D_possible_values = np.logspace(np.log10(MIN_D), np.log10(MAX_D), num=10000)

        """
        If boundary effects are ignored, we set a relative high L value
        """
        if not ignore_boundary_effects:
            custom_dic['L'] = L
        else:
            if model_label in [1,2]:
                custom_dic['L'] = None
            else:
                custom_dic['L'] = 1024

        """
        We set D and alpha for models 1 and 3
        """
        if model_label in [1,3]:
            D = np.random.choice(D_possible_values)
            ALPHA = models_phenom().bound_alpha[1] if force_directed else np.random.choice(ALPHA_possible_values)
            custom_dic.update({ 'Ds': [D, D*0.01], 'alphas': [ALPHA, 0.01]})

        """
        For model 2, transition matrix is created between
        2 and 5 different states.
        """
        if model_label == 2:
            n = np.random.randint(2,5)
            transition_matrix = np.zeros((n,n))
            p = np.random.uniform(0.80,1)
            for i in range(n):
                for j in range(n):
                    transition_matrix[i, j] = p if i==j else (1-p)/(n-1)

            similar_order_magnitude = np.random.choice([False, True])
            if similar_order_magnitude:
                ds_values = [np.random.choice(D_possible_values)] + [None] * (n-1)
                magnitude_order = int(np.log10(ds_values[0]))
                minimum_magnitude_order = max(magnitude_order-1,-12)
                maximum_magnitude_order = min(magnitude_order+1,6)
                for ds_value_i in range(1,n):
                    ds_values[ds_value_i] = 10**np.random.uniform(minimum_magnitude_order,maximum_magnitude_order)
            else:
                ds_values = np.random.choice(D_possible_values, size=n, replace=False)

            as_values = np.random.choice(ALPHA_possible_values, size=n, replace=False)

            custom_dic.update({
                'model': datasets_phenom().avail_models_name[1],
                'M': transition_matrix, #transition matrix
                'return_state_num': True,
                'Ds': np.array([[d, d*0.01] for d in ds_values]),
                'alphas': np.array([[a, a*0.01] for a in as_values]),
            })

        if model_label in [4,5]:
            fast_D = np.random.choice(D_possible_values)

            similar_order_magnitude = np.random.choice([False, True])
            if similar_order_magnitude:
                magnitude_order = int(np.log10(fast_D))
                minimum_magnitude_order = max(magnitude_order-1,-12)
                maximum_magnitude_order = np.log10(fast_D)
                slow_D = 10**np.random.uniform(minimum_magnitude_order,maximum_magnitude_order)
            else:
                slow_D = np.random.choice(D_possible_values[D_possible_values<=fast_D])

            assert slow_D <= fast_D
            alpha1 = models_phenom().bound_alpha[1] if force_directed else np.random.choice(ALPHA_possible_values)
            alpha2 = np.random.uniform(MIN_A, 1.8)

            custom_dic.update({
                'Ds': np.array([[fast_D, fast_D*0.01], [slow_D, slow_D*0.01]]),
                'alphas': np.array([[alpha1, 0.01], [alpha2, 0.01]])
            })

        custom_dic.update({'model': datasets_phenom().avail_models_name[model_label-1]})

        if model_label in [3,4]:
            custom_dic.update({
                'Pu': np.random.uniform(0,0.05), # Unbinding probability
                'Pb': np.random.uniform(0.95,1.00)  # Binding probability
            })

        if model_label == 3:
            custom_dic.update({
                'Nt': int((custom_dic['L']**2)*TRAP_DENSITY), # Number of traps
                'r': np.random.uniform(0.5,1.0)
            })

        if model_label == 4:
            custom_dic.update({
                'r': np.random.uniform(0.5,1.0), # Size of particles
                'return_state_num': True         # To get the state numeration back, hence labels.shape = TxNx4
            })

        if model_label == 5:
            custom_dic.update({
                'r': np.random.uniform(5,20),
                'Nc': int((custom_dic['L']**2)*CONFINEMENTS_DENSITY),
                'trans':np.random.uniform(0,0.50)
            })

        dic = _get_dic_andi2(model_label)
        dic['T'] = trajectory_length
        dic['N'] = number_of_trajectories if number_of_trajectories is not None else int((custom_dic['L']**2)*TRAJECTORY_DENSITY)

        for key in custom_dic:
            dic[key] = custom_dic[key]
        #print(dic)
        return dic

    def get_trayectories_from_file(self, file_name, limit=float('inf')):
        trajectories = []
        dataframe = pd.read_csv(file_name)
        for i, unique_id in enumerate(dataframe['id'].unique()):
            if i > limit-1:
                break
            t_dataframe = dataframe[dataframe['id'] == unique_id]
            trajectories.append(Trajectory(
                x=t_dataframe['x'].tolist(),
                y=t_dataframe['y'].tolist(),
                t=t_dataframe['t'].tolist(),
                info={
                    'alpha_t': t_dataframe['alpha_t'].tolist(),
                    'd_t': t_dataframe['d_t'].tolist(),
                    'state_t': t_dataframe['state_t'].tolist()
                },
            ))
        return trajectories

    def save_trajectories(self, trajectories, file_name):
        data = {
            'id':[],
            't':[],
            'x':[],
            'y':[],
            'd_t':[],
            'alpha_t':[],
            'state_t':[]
        }

        for i, t in enumerate(trajectories):
            data['id'] += [i] * t.length
            data['t'] += t.get_time().tolist()
            data['x'] += t.get_x().tolist()
            data['y'] += t.get_y().tolist()
            data['d_t'] += list(t.info['d_t'])
            data['alpha_t'] += list(t.info['alpha_t'])
            data['state_t'] += list(t.info['state_t'])

        pd.DataFrame(data).to_csv(file_name, index=False)

    def simulate_phenomenological_trajectories_for_regression_training(
            self,
            number_of_trajectories,
            trajectory_length,
            trajectory_time, # For MINFLUX I'm going to modify this variable
            get_from_cache=False,
            file_label='',
            ignore_boundary_effects=True,
            read_limit=float('inf')
        ):
        FILE_NAME = f't_{file_label}_{trajectory_length}_{trajectory_time}_{number_of_trajectories}_boundary_{ignore_boundary_effects}_regression.cache'

        if get_from_cache and os.path.exists(FILE_NAME):
            trajectories = self.get_trayectories_from_file(FILE_NAME, limit=read_limit)
        else:
            trajectories = []
            with tqdm.tqdm(total=number_of_trajectories) as pbar:
                while len(trajectories) < number_of_trajectories:
                    sim_dic = self.__generate_dict_for_model(2, trajectory_length, 5, ignore_boundary_effects=True)

                    def include_trajectory(trajectory): #We want a diverse number of characteristics
                        segments_lengths = np.diff(np.where(np.diff(trajectory.info['d_t']) != 0))
                        return len(np.unique(trajectory.info['d_t'])) > 1 and trajectory.length == trajectory_length and not np.any(segments_lengths < 3)

                    sim_dic.pop('model')
                    trajs, labels = models_phenom().multi_state(**sim_dic)
                    new_list_of_trajectories = [ti for ti in Trajectory.from_models_phenom(trajs, labels) if include_trajectory(ti)][:2]
                    trajectories += new_list_of_trajectories
                    pbar.update(len(new_list_of_trajectories))

            np.random.shuffle(trajectories)
            trajectories = trajectories[:number_of_trajectories]

            if get_from_cache:
                self.save_trajectories(trajectories, FILE_NAME)

        return trajectories

    def simulate_phenomenological_trajectories_for_classification_training(self, number_of_trajectories, trajectory_length, trajectory_time, get_from_cache=False, file_label='', type_of_simulation='models_phenom', ignore_boundary_effects=True, enable_parallelism=False, read_limit=float('inf')):
        FILE_NAME = f't_{file_label}_{trajectory_length}_{trajectory_time}_{number_of_trajectories}_mode_{type_of_simulation}_classification.cache'
        if get_from_cache and os.path.exists(FILE_NAME):
            trajectories = self.get_trayectories_from_file(FILE_NAME, limit=read_limit)
        else:
            trajectories = []
            with tqdm.tqdm(total=number_of_trajectories) as pbar:
                def generate_trayectory(limit):
                    parameter_simulation_setup = [
                        {'model': 1, 'force_directed': False},
                        {'model': 2, 'force_directed': False},
                        #{'model': 3, 'force_directed': False},
                        {'model': 4, 'force_directed': False},
                    ]

                    simulation_setup = np.random.choice(parameter_simulation_setup, p=[0.25, (0.75)/2, (0.75)/2])
                    retry = True
                    while retry:
                        dic = self.__generate_dict_for_model(simulation_setup['model']+1, trajectory_length, 50, force_directed=simulation_setup['force_directed'], ignore_boundary_effects=ignore_boundary_effects, L=512)

                        def include_trajectory(trajectory): #We want a diverse number of characteristics
                            segments_lengths = np.diff(np.where(np.diff(trajectory.info['d_t']) != 0))
                            x_diff = np.max(trajectory.get_noisy_x()) - np.min(trajectory.get_noisy_x())
                            y_diff = np.max(trajectory.get_noisy_y()) - np.min(trajectory.get_noisy_y())
                            within_roi = x_diff <= 512 and y_diff <= 512 if not ignore_boundary_effects else x_diff <= 1000 and y_diff <= 1000
                            within_roi = within_roi or np.all(trajectory.info['state_t'] == 2)
                            return within_roi and len(np.unique(trajectory.info['d_t'])) > 1 and trajectory.length == trajectory_length and not np.any(segments_lengths < 3)
                        new_trajectories = []
                        if type_of_simulation == 'create_dataset':
                            trajs, labels = datasets_phenom().create_dataset(dics = dic)
                            new_trajectories += [ti for ti in Trajectory.from_datasets_phenom(trajs, labels) if include_trajectory(ti)]
                        elif type_of_simulation == 'challenge_phenom_dataset':
                            try:
                                trajs, labels, _ = challenge_phenom_dataset(experiments = 1, num_fovs = 5, dics = [dic], repeat_exp=False)
                                new_trajectories += [ti for ti in Trajectory.from_challenge_phenom_dataset(trajs, labels) if include_trajectory(ti)]
                            except ValueError:
                                pass
                        elif type_of_simulation == 'models_phenom':
                            function_to_use = {
                                1: models_phenom().single_state,
                                2: models_phenom().multi_state,
                                3: models_phenom().immobile_traps,
                                4: models_phenom().dimerization,
                                5: models_phenom().confinement
                            }[simulation_setup['model']+1]

                            dic.pop('model')
                            trajs, labels = function_to_use(**dic)
                            new_trajectories += [ti for ti in Trajectory.from_models_phenom(trajs, labels) if include_trajectory(ti)]
                        else:
                            raise Exception(f'type_of_simulation={type_of_simulation} is not possible')
                        if len(new_trajectories) > 0:
                            new_trajectories = new_trajectories[:limit]
                            retry = False
                    return new_trajectories
 
                if enable_parallelism:
                    @ray.remote
                    def generate_trayectory_to_use(limit):
                        return generate_trayectory(limit)
                    ray.init()
                    while len(trajectories) < number_of_trajectories:
                        new_list_of_trajectories = ray.get([generate_trayectory_to_use.remote(5) for _ in range(100)])
                        new_trajectories = []
                        for t_list in new_list_of_trajectories:
                            new_trajectories += t_list
                        trajectories += new_trajectories
                        pbar.update(len(new_trajectories))
                    ray.shutdown()
                else:
                    while len(trajectories) < number_of_trajectories:
                        new_trajectories = generate_trayectory(5)
                        trajectories += new_trajectories
                        pbar.update(len(new_trajectories))

            np.random.shuffle(trajectories)
            trajectories = trajectories[:number_of_trajectories]

            if get_from_cache:
                self.save_trajectories(trajectories, FILE_NAME)
        return trajectories

    def simulate_challenge_trajectories(self, number_of_exps, filter=False):
        parameter_simulation_setup = [
            {'model': 0, 'force_directed': False},
            {'model': 1, 'force_directed': False},
            {'model': 2, 'force_directed': False},
            #{'model': 3, 'force_directed': False},
            {'model': 4, 'force_directed': False},
        ]

        for exp_index in range(number_of_exps):
            retry = True
            while retry:
                try:
                    simulation_setup = np.random.choice(parameter_simulation_setup)
                    simulation_dic = self.__generate_dict_for_model(simulation_setup['model']+1, 200, None, ignore_boundary_effects=False, L=1024)
                    dfs_traj, labs_traj, _ = challenge_phenom_dataset(
                        experiments = 1,
                        save_data = True,
                        dics = [simulation_dic],
                        return_timestep_labs = True, get_video = False,
                        num_fovs = 5,
                        path='./2ndAndiTrajectories_Challenge_Test/',
                        prefix=str(exp_index)
                    )
                    retry = False
                except ValueError as e:
                    print(f"Simulation failed: {e}")
                    for file_path in glob.glob(f'./2ndAndiTrajectories_Challenge_Test/{exp_index}*'):
                        os.remove(file_path)
                    retry = True

        #trajs, labels, _ = challenge_phenom_dataset(experiments = 1, num_fovs = 1, dics = [dic], repeat_exp=False)
        #trajectories = Trajectory.from_challenge_phenom_dataset(dfs_traj, labs_traj)
        #if simulation_setup['model'] >= 2 and filter:
        #    trajectories = [t for t in trajectories if len(np.unique(t.info['state_t'])) > 1]
        #return trajectories
