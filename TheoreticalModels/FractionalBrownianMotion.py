import numpy as np

from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time
from CONSTANTS import EXPERIMENT_PIXEL_SIZE, IGNORE_MULTI_MODEL_CLASSIFICATION
from TheoreticalModels.Model import Model
from andi_datasets.models_phenom import models_phenom

class FractionalBrownianMotion(Model):
    STRING_LABEL = 'fbm'

    SUB_DIFFUSIVE_HURST_EXPONENT_RANGE = [0.05/2, 0.95/2]
    SUP_DIFFUSIVE_HURST_EXPONENT_RANGE = [1.05/2, 1.95/2]
    NOT_EXACT_BROWNIAN_HURST_EXPONENT_RANGE = [SUB_DIFFUSIVE_HURST_EXPONENT_RANGE[1], SUP_DIFFUSIVE_HURST_EXPONENT_RANGE[0]]
    D_RANGE = [0.001, 1] #micrometer^2/s

    @classmethod
    def create_random_instance(cls):
        if not IGNORE_MULTI_MODEL_CLASSIFICATION:
            selected_diffusion = np.random.choice(
                ['subdiffusive', 'brownian', 'superdiffusive'])
            if selected_diffusion == 'superdiffusive':
                selected_range = cls.SUP_DIFFUSIVE_HURST_EXPONENT_RANGE
            elif selected_diffusion == 'subdiffusive':
                selected_range = cls.SUB_DIFFUSIVE_HURST_EXPONENT_RANGE
            elif selected_diffusion == 'brownian':
                selected_range = cls.NOT_EXACT_BROWNIAN_HURST_EXPONENT_RANGE

            selected_hurst_exponent = np.random.uniform(selected_range[0], selected_range[1])
        else:
            selected_hurst_exponent = np.random.uniform(cls.SUB_DIFFUSIVE_HURST_EXPONENT_RANGE[0], cls.SUP_DIFFUSIVE_HURST_EXPONENT_RANGE[1])
        
        selected_diffusion_coefficient = np.random.choice(np.logspace(np.log10(cls.D_RANGE[0]), np.log10(cls.D_RANGE[1]), 1000))

        return cls(hurst_exponent=selected_hurst_exponent, diffusion_coefficient=selected_diffusion_coefficient)

    @classmethod
    def string_label(cls):
        return cls.STRING_LABEL

    def __init__(self, hurst_exponent=None, diffusion_coefficient=None):
        self.hurst_exponent = hurst_exponent
        self.diffusion_coefficient = diffusion_coefficient

    @property
    def anomalous_exponent(self):
        return self.hurst_exponent * 2

    """
    Previous Works from ourt laboratory worked with the simulation presented on this paper:

    Granik N, Weiss LE, Nehme E, Levin M, Chein M, Perlson E, Roichman Y, Shechtman Y. 
    Single-Particle Diffusion Characterization by Deep Learning. 
    Biophys J. 2019 Jul 23;117(2):185-192. doi: 10.1016/j.bpj.2019.06.015. Epub 2019 Jun 22. 
    PMID: 31280841; PMCID: PMC6701009.
    
    Original code: https://github.com/AnomDiffDB/DB/blob/master/utils.py

    During the development of the new Andi-Challenge competition, a single-state fBM simulation
    was developed that enables two parameters as input: anomalous exponent and diffusion
    coefficient. This time We use that simulation.
    """
    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        micrometer_per_pixel = EXPERIMENT_PIXEL_SIZE / 1000
        delta_t = trajectory_time / trajectory_length

        unidimensional_diffusion_coefficient = self.diffusion_coefficient * ((delta_t)/(micrometer_per_pixel**2)) #micrometer^2/s -> pixel^2/frame

        raw_trajectories, _ = models_phenom().single_state(
            N = 1,
            L=10/(EXPERIMENT_PIXEL_SIZE/1000),
            T = trajectory_length,
            Ds = unidimensional_diffusion_coefficient,
            alphas = self.anomalous_exponent
            )

        x,y = raw_trajectories[:,0,0] * EXPERIMENT_PIXEL_SIZE, raw_trajectories[:,0,1] * EXPERIMENT_PIXEL_SIZE
        t = simulate_track_time(trajectory_length, trajectory_time)

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': 'hurst',
            'exponent': self.hurst_exponent,
            'info': {'diffusion_coefficient': self.diffusion_coefficient}
        }

class FractionalBrownianMotionSuperDiffusive(FractionalBrownianMotion):
    STRING_LABEL = 'fbm_sup'

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.SUP_DIFFUSIVE_HURST_EXPONENT_RANGE
        selected_hurst_exponent = np.random.uniform(selected_range[0], selected_range[1])
        selected_diffusion_coefficient = np.random.uniform(cls.D_RANGE[0], cls.D_RANGE[1])
        return cls(hurst_exponent=selected_hurst_exponent, diffusion_coefficient=selected_diffusion_coefficient)

class FractionalBrownianMotionSubDiffusive(FractionalBrownianMotion):
    STRING_LABEL = 'fbm_sub'

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.SUB_DIFFUSIVE_HURST_EXPONENT_RANGE
        selected_hurst_exponent = np.random.uniform(selected_range[0], selected_range[1])
        selected_diffusion_coefficient = np.random.uniform(cls.D_RANGE[0], cls.D_RANGE[1])
        return cls(hurst_exponent=selected_hurst_exponent, diffusion_coefficient=selected_diffusion_coefficient)

class FractionalBrownianMotionBrownian(FractionalBrownianMotion):
    STRING_LABEL = 'fbm_brownian'

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.NOT_EXACT_BROWNIAN_HURST_EXPONENT_RANGE
        selected_hurst_exponent = np.random.uniform(selected_range[0], selected_range[1])
        selected_diffusion_coefficient = np.random.uniform(cls.D_RANGE[0], cls.D_RANGE[1])
        return cls(hurst_exponent=selected_hurst_exponent, diffusion_coefficient=selected_diffusion_coefficient)
