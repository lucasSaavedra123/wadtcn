from random import shuffle
from numpy.random import choice


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
