from contextlib import redirect_stdout
import io

import numpy as np

from Trajectory import Trajectory
from CONSTANTS import EXPERIMENT_HEIGHT, EXPERIMENT_WIDTH, IMMOBILE_THRESHOLD
from andi_datasets.datasets_theory  import datasets_theory
#from andi_datasets.datasets_challenge import challenge_theory_dataset #Slow Function


def normalize(trajs, variance=None):    
    '''
    Normalizes trajectories by substracting average and dividing by
    SQRT of their standard deviation.
    
    Parameters
    ----------
    trajs : np.array
        Array of length N x T or just T containing the ensemble or single trajectory to normalize. 
    '''
    # Checking and saving initial shape
    initial_shape = trajs.shape
    if len(trajs.shape) == 1: # single one d trajectory
        trajs = trajs.reshape(1, trajs.shape[0], 1)
    if len(trajs.shape) == 2: # ensemble of one d trajectories
        trajs = trajs.reshape(trajs.shape[0], trajs.shape[1], 1)
        
    trajs = trajs - trajs.mean(axis=1, keepdims=True)
    displacements = (trajs[:,1:,:] - trajs[:,:-1,:]).copy()    
    variance = np.std(displacements, axis=1)
    variance[variance == 0] = 1 
    new_trajs = np.cumsum((displacements/np.expand_dims(variance, axis = 1)), axis = 1)
    initial_zeros = np.expand_dims(np.zeros((new_trajs.shape[0], new_trajs.shape[-1])), axis = 1)
    return np.concatenate((initial_zeros, new_trajs), axis = 1).reshape(initial_shape)

class Model():
    STRING_LABELS = ['ctrw', 'fbm', 'od', 'lw', 'attm', 'sbm', 'bm']
    MODEL_COLORS = {
        'ctrw': 'blue',
        'fbm': 'red',
        'od': 'black',
        'lw': 'orange',
        'attm': 'grey',
        'sbm': 'purple',
        'bm': 'cyan'
    }

    @classmethod
    def create_random_instance(cls):
        raise Exception('create_random_instance class method should be implemented')

    @classmethod
    def string_label(cls):
        return cls.STRING_LABEL

    @classmethod
    def model_color(cls):
        return cls.MODEL_COLORS[cls.STRING_LABEL]

    @classmethod
    def numeric_label(cls):
        return cls.STRING_LABELS.index(cls.string_label())

    def simulate_trajectory(self, trajectory_length, trajectory_time, from_andi=False):
        resimulate = True

        while resimulate:
            if from_andi:
                simulation_result = self.andi_simulate_rawly(trajectory_length, trajectory_time)
                resimulate = False
                
                trajectory = Trajectory(
                    simulation_result['x'],
                    simulation_result['y'],
                    t=simulation_result['t'],
                    #noise_x=simulation_result['x_noisy']-simulation_result['x'],
                    #noise_y=simulation_result['y_noisy']-simulation_result['y'],
                    #exponent_type=simulation_result['exponent_type'],
                    #exponent=simulation_result['exponent'],
                    model_category=self,
                    info=simulation_result['info'],
                    noisy=True
                )
            else:
                simulation_result = self.custom_simulate_rawly(trajectory_length, trajectory_time)

                trajectory = Trajectory(
                    simulation_result['x'],
                    simulation_result['y'],
                    t=simulation_result['t'],
                    noise_x=simulation_result['x_noisy']-simulation_result['x'],
                    noise_y=simulation_result['y_noisy']-simulation_result['y'],
                    exponent_type=simulation_result['exponent_type'],
                    exponent=simulation_result['exponent'],
                    model_category=self,
                    info=simulation_result['info']
                )

                check_one = (min(simulation_result['x']) < 0 or min(simulation_result['x_noisy']) < 0 or max(simulation_result['x']) > EXPERIMENT_WIDTH or max(simulation_result['x_noisy']) > EXPERIMENT_WIDTH)
                check_two = (min(simulation_result['y']) < 0 or min(simulation_result['y_noisy']) < 0 or max(simulation_result['y']) > EXPERIMENT_HEIGHT or max(simulation_result['y_noisy']) > EXPERIMENT_HEIGHT)
                check_three = ('switching' in simulation_result['info'] and not simulation_result['info']['switching'])
                check_four = trajectory.is_immobile(IMMOBILE_THRESHOLD)
                resimulate = any([check_one, check_two, check_three, check_four])

        return trajectory

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        raise Exception('custom_simulate_rawly method should be implemented')

    def __str__(self):
        return self.STRING_LABEL

    def andi_simulate_rawly(self, trajectory_length, trajectory_time):
        model_number = datasets_theory().avail_models_name.index(self.STRING_LABEL)

        retry = True

        #with redirect_stdout(io.StringIO()):
        """
        This part should be changes because is too slow
        """

        """
        while retry:
            _,_,X,Y,_,_ = challenge_theory_dataset(1, min_T=trajectory_length, max_T=trajectory_length+1, tasks=2, dimensions=2)
            retry = Y[1][0] != model_number
        """

        """
        The following piece of code is part of the ANDI Challenge Code
        """

        dim = 2
        max_T = trajectory_length

        dataset = datasets_theory().create_dataset(
            T = max_T,
            N_models = 1,
            exponents = [self.anomalous_exponent], 
            dimension = dim,
            models = [datasets_theory().avail_models_name.index(self.__class__.STRING_LABEL.split('_')[0])]
        )

        # Normalize trajectories
        n_traj = dataset.shape[0]
        norm_trajs = normalize(dataset[:, 2:].reshape(n_traj*dim, max_T))
        dataset[:, 2:] = norm_trajs.reshape(dataset[:, 2:].shape)

        # Save unnoisy dataset for task3
        dataset_t3 = dataset.copy()

        # Add localization error, Gaussian noise with sigma = [0.1, 0.5, 1]                
        loc_error_amplitude = np.random.choice(np.array([0.1, 0.5, 1]), size = n_traj).repeat(dim)
        loc_error = (np.random.randn(n_traj*dim, int(max_T)).transpose()*loc_error_amplitude).transpose()                        
        dataset = datasets_theory().create_noisy_localization_dataset(dataset, dimension = dim, T = max_T, noise_func = loc_error)

        # Add random diffusion coefficients            
        trajs = dataset[:, 2:].reshape(n_traj*dim, max_T)
        displacements = trajs[:, 1:] - trajs[:, :-1]
        # Get new diffusion coefficients and displacements
        diffusion_coefficients = np.random.randn(n_traj).repeat(dim)
        new_displacements = (displacements.transpose()*diffusion_coefficients).transpose()  
        # Generate new trajectories and add to dataset
        new_trajs = np.cumsum(new_displacements, axis = 1)
        new_trajs = np.concatenate((np.zeros((new_trajs.shape[0], 1)), new_trajs), axis = 1)
        dataset[:, 2:] = new_trajs.reshape(dataset[:, 2:].shape)

        #trajectory_raw = X[1][0]
        #x = trajectory_raw[:trajectory_length]
        #y = trajectory_raw[trajectory_length:]

        x = dataset[0][2:trajectory_length+2]
        y = dataset[0][trajectory_length+2:]

        return {
            'x': x,
            'y': y,
            't': np.arange(0,trajectory_length,1)*trajectory_time/trajectory_length,
            #'x_noisy': noisy_x,
            #'y_noisy': noisy_y,
            'exponent_type': 'anomalous',
            'exponent': self.anomalous_exponent,
            'info': {}
        }
