from os.path import join

import numpy as np
from keras.models import load_model


from CONSTANTS import *
from .model_utils import transform_trajectories_to_categorical_vector
from .randi_utils import *
from .PredictiveModel import PredictiveModel

class LSTMTheoreticalModelSwitchTimeAndExponent(PredictiveModel):

    @classmethod
    def selected_hyperparameters(self):
        return {'None': 'None'}

    def default_hyperparameters(self):
        return {'None': 'None'}

    @classmethod
    def default_hyperparameters_analysis(self):
        return None

    """
    This segmentation and data transformation comes from paper:

    Aykut Argun, Giovanni Volpe, Stefano Bo

    Classification, Inference, and Segmentation of anomalous
    diffusion with recurrent neural networks

    Original code: https://github.com/argunaykut/randi/blob/main/using_the_nets.ipynb
    """

    def predict(self, trajectories):
        inf_seg = self.architecture.predict(self.transform_trajectories_to_input(trajectories))
        pred_t = my_atan(inf_seg[:,2],inf_seg[:,3])*200/(2*np.pi)

        first_exponents = inf_seg[:,0]
        second_exponents = inf_seg[:,1]
        switching_times = pred_t
        return [first_exponents, second_exponents], switching_times

    def transform_trajectories_to_input(self, trajectories):
        N = len(trajectories)
        dimension = 2
        traj_length = trajectories[0].length

        X = np.zeros((N, traj_length * dimension))

        for i in range(len(trajectories)):
            X[i,:traj_length] = trajectories[i].get_noisy_x()
            X[i,traj_length:] = trajectories[i].get_noisy_y()

        block_size = self.architecture.layers[0].input_shape[-1]

        data = data_norm(X,dim=2,task=3)
        data_rs = data_reshape(data,bs=block_size,dim=2)
        return data_rs

    @property
    def type_name(self):
        return f"randi_segmentation_inf"

    def __str__(self):
        return self.type_name

    def save_as_file(self):
        raise Exception("RANDI is not persisted in MongoDB")

    def load_as_file(self):
        self.architecture = load_model(join(NETWORKS_DIRECTORY, f"{str(self)}.h5"))
