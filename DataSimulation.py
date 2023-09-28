from random import shuffle
from numpy.random import choice
import numpy as np
from andi_datasets.datasets_challenge import challenge_theory_dataset
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