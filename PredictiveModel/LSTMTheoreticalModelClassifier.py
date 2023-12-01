from os.path import join

import numpy as np
from keras.models import load_model


from CONSTANTS import *
from .model_utils import transform_trajectories_to_categorical_vector
from .randi_utils import *
from .PredictiveModel import PredictiveModel

class LSTMTheoreticalModelClassifier(PredictiveModel):

    @classmethod
    def selected_hyperparameters(self):
        return {'None': 'None'}

    def default_hyperparameters(self):
        return {'None': 'None'}

    @classmethod
    def default_hyperparameters_analysis(self):
        return None

    """
    This classification comes from paper:

    Aykut Argun, Giovanni Volpe, Stefano Bo

    Classification, Inference, and Segmentation of anomalous
    diffusion with recurrent neural networks

    Original code: https://github.com/argunaykut/randi/blob/main/using_the_nets.ipynb
    """
    @classmethod
    def classify_with_combination(cls, trajectories, list_of_classifiers):
        N = len(trajectories)
        dimension = 2
        traj_length = trajectories[0].length

        X = np.zeros((N, traj_length * dimension))

        for i in range(N):
            X[i,:traj_length] = trajectories[i].get_noisy_x()
            X[i,traj_length:] = trajectories[i].get_noisy_y()

        centers_class_2d = [classifier.trajectory_length for classifier in list_of_classifiers]
        meta_model_class_2d = [classifier.architecture for classifier in list_of_classifiers]

        Y_predicted = many_net_uhd(nets = meta_model_class_2d, traj_set = X, centers = centers_class_2d ,dim = 2, task =2).reshape(-1,5)
        Y_predicted = np.argmax(Y_predicted, axis=-1)

        return Y_predicted

    """
    This transformation comes from paper:

    Aykut Argun, Giovanni Volpe, Stefano Bo

    Classification, Inference, and Segmentation of anomalous
    diffusion with recurrent neural networks

    Original code: https://github.com/argunaykut/randi/blob/main/using_the_nets.ipynb
    """
    def transform_trajectories_to_input(self, trajectories):
        N = len(trajectories)
        dimension = 2
        traj_length = trajectories[0].length

        X = np.zeros((N, traj_length * dimension))

        for i in range(len(trajectories)):
            X[i,:traj_length] = trajectories[i].get_noisy_x()
            X[i,traj_length:] = trajectories[i].get_noisy_y()

        block_size = self.architecture.layers[0].input_shape[-1]

        data = data_norm(X,dim=2,task=2)
        data_rs = data_reshape(data,bs=block_size,dim=2)

        return data_rs

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_categorical_vector(self, trajectories)

    @property
    def type_name(self):
        return f"randi_classification"

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}"

    def save_as_file(self):
        raise Exception("RANDI is not persisted in MongoDB")

    def load_as_file(self):
        self.architecture = load_model(join(NETWORKS_DIRECTORY, f"{str(self)}.h5"))
