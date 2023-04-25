import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset


class LevyWalk(Model):
    STRING_LABEL = 'lw'
    ANOMALOUS_EXPONENT_RANGE = [1.1, 1.9]
    VELOCITY_RANGE = [0.002, 0.005] #um/ms

    @classmethod
    def create_random_instance(cls):
        anomalous_exponent = np.random.uniform(low=cls.ANOMALOUS_EXPONENT_RANGE[0], high=cls.ANOMALOUS_EXPONENT_RANGE[1])
        velocity = np.random.uniform(low=cls.VELOCITY_RANGE[0], high=cls.VELOCITY_RANGE[1])
        model = cls(anomalous_exponent=anomalous_exponent, velocity=velocity)
        return model

    def __init__(self, anomalous_exponent, velocity):
        self.anomalous_exponent = anomalous_exponent
        self.velocity = velocity * 1000 # to um/s

    """
    This simulation comes from paper:

    Muñoz-Gil, G., Volpe, G., Garcia-March, M.A. et al.
    Objective comparison of methods to decode anomalous diffusion.
    Nat Commun 12, 6253 (2021). https://doi.org/10.1038/s41467-021-26320-w

    Original Code: https://github.com/AnDiChallenge/andi_datasets/blob/master/functions/diffusion_models.py
    """
    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        if self.anomalous_exponent < 1:
            raise ValueError(
                'Levy walks only allow for anomalous exponents > 1.')
        # Define exponents for the distribution of times
        if self.anomalous_exponent == 2:
            sigma = np.random.rand()
        else:
            sigma = 3-self.anomalous_exponent
        dt = (1-np.random.rand(trajectory_length))**(-1/sigma)
        dt[dt > trajectory_length] = trajectory_length+1
        time_per_step = trajectory_time/trajectory_length
        # Define the velocity
        #v = 10*np.random.rand()
        #v = np.sqrt(self.diffusion_coefficient*time_per_step*2)/(time_per_step)
        v = self.velocity
        # Define the array where we save step length
        d = np.empty(0)
        # Define the array where we save the angle of the step
        angles = np.empty(0)
        # Generate trajectory
        for t in dt:
            r = np.random.choice([-1,1])
            new_step_lengths = v*np.ones(int(t))*time_per_step*r
            d = np.append(d, new_step_lengths)
            angles = np.append(angles, np.random.uniform(low=0, high=2*np.pi)*np.ones(int(t)))
            if len(d) > trajectory_length:
                break
        d = d[:int(trajectory_length)]
        angles = angles[:int(trajectory_length)]
        posX, posY = [d*np.cos(angles), d*np.sin(angles)]
        #return np.concatenate((np.cumsum(posX)-posX[0], np.cumsum(posY)-posY[0]))

        x = (np.cumsum(posX)-posX[0]) * 1000 #to nm
        y = (np.cumsum(posY)-posY[0]) * 1000 #to nm
        t = np.arange(0,trajectory_length,1)*trajectory_time/trajectory_length

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)
        t = np.arange(0, trajectory_length, 1) * time_per_step
        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': 'anomalous',
            'exponent': self.anomalous_exponent,
            'info': {}
        }
