import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time
from scipy.special import erfcinv
from CONSTANTS import EXPERIMENT_PIXEL_SIZE

class ScaledBrownianMotion(Model):
    STRING_LABEL = 'sbm'
    
    SUB_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE = [0.05, 0.95]
    SUP_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE = [1.05, 1.95]
    NOT_EXACT_BROWNIAN_ANOMALOUS_EXPONENT_RANGE = [SUB_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE[1], SUP_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE[0]]

    @classmethod
    def create_random_instance(cls):
        selected_diffusion = np.random.choice(
            ['subdiffusive', 'brownian', 'superdiffusive'])
        if selected_diffusion == 'superdiffusive':
            selected_range = cls.SUP_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE
        elif selected_diffusion == 'subdiffusive':
            selected_range = cls.SUB_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE
        elif selected_diffusion == 'brownian':
            selected_range = cls.NOT_EXACT_BROWNIAN_ANOMALOUS_EXPONENT_RANGE

        selected_anomalous_exponent = np.random.uniform(
            selected_range[0], selected_range[1])

        return cls(anomalous_exponent=selected_anomalous_exponent)

    def __init__(self, anomalous_exponent):
        self.anomalous_exponent = anomalous_exponent

    """
    This simulation comes from paper:

    Muñoz-Gil, G., Volpe, G., Garcia-March, M.A. et al.
    Objective comparison of methods to decode anomalous diffusion.
    Nat Commun 12, 6253 (2021). https://doi.org/10.1038/s41467-021-26320-w

    Original Code: https://github.com/AnDiChallenge/andi_datasets/blob/master/functions/diffusion_models.py
    """
    def custom_simulate_rawly(self, trajectory_length, trajectory_time, sigma=1):
        '''Creates a scaled brownian motion trajectory'''
        t = simulate_track_time(trajectory_length+1, trajectory_time)
        msd = (sigma**2)*np.arange(trajectory_length+1)**self.anomalous_exponent
        deltas = np.sqrt(msd[1:]-msd[:-1])
        dx = np.sqrt(2)*deltas*erfcinv(2-2*np.random.rand(len(deltas)))            
        dy = np.sqrt(2)*deltas*erfcinv(2-2*np.random.rand(len(deltas)))  
        posX = np.cumsum(dx)-dx[0]
        posY = np.cumsum(dy)-dy[0]

        x = posX[:trajectory_length] * (EXPERIMENT_PIXEL_SIZE)#* 1000 #to nm
        y = posY[:trajectory_length] * (EXPERIMENT_PIXEL_SIZE)#* 1000 #to nm

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)
        t = t[:trajectory_length]

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

class ScaledBrownianMotionSuperDiffusive(ScaledBrownianMotion):
    STRING_LABEL = "sbm_sup"

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.SUP_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE

        selected_anomalous_exponent = np.random.uniform(
            selected_range[0], selected_range[1])

        return cls(anomalous_exponent=selected_anomalous_exponent)


class ScaledBrownianMotionSubDiffusive(ScaledBrownianMotion):
    STRING_LABEL = "sbm_sub"

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.SUB_DIFFUSIVE_ANOMALOUS_EXPONENT_RANGE
        selected_anomalous_exponent = np.random.uniform(selected_range[0], selected_range[1])
        return cls(anomalous_exponent=selected_anomalous_exponent)


class ScaledBrownianMotionBrownian(ScaledBrownianMotion):
    STRING_LABEL = "sbm_brownian"

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.NOT_EXACT_BROWNIAN_ANOMALOUS_EXPONENT_RANGE
        selected_anomalous_exponent = np.random.uniform(selected_range[0], selected_range[1])
        return cls(anomalous_exponent=selected_anomalous_exponent)
