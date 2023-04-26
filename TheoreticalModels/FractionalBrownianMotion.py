import numpy as np
from scipy import fftpack

from TheoreticalModels.simulation_utils import add_noise_and_offset
from CONSTANTS import EXPERIMENT_PIXEL_SIZE
from TheoreticalModels.Model import Model


class FractionalBrownianMotion(Model):
    STRING_LABEL = 'fbm'
    SUB_DIFFUSIVE_HURST_EXPONENT_RANGE = [0.1, 0.42]
    SUP_DIFFUSIVE_HURST_EXPONENT_RANGE = [0.58, 0.9]
    NOT_EXACT_BROWNIAN_HURST_EXPONENT_RANGE = [SUB_DIFFUSIVE_HURST_EXPONENT_RANGE[1], SUP_DIFFUSIVE_HURST_EXPONENT_RANGE[0]]

    @classmethod
    def create_random_instance(cls):
        selected_diffusion = np.random.choice(
            ['subdiffusive', 'brownian', 'superdiffusive'])
        if selected_diffusion == 'superdiffusive':
            selected_range = cls.SUP_DIFFUSIVE_HURST_EXPONENT_RANGE
        elif selected_diffusion == 'subdiffusive':
            selected_range = cls.SUB_DIFFUSIVE_HURST_EXPONENT_RANGE
        elif selected_diffusion == 'brownian':
            selected_range = cls.NOT_EXACT_BROWNIAN_HURST_EXPONENT_RANGE

        selected_hurst_exponent = np.random.uniform(
            selected_range[0], selected_range[1])

        return cls(hurst_exponent=selected_hurst_exponent)

    @classmethod
    def string_label(cls):
        return cls.STRING_LABEL

    def __init__(self, hurst_exponent=None):
        self.hurst_exponent = hurst_exponent

    @property
    def anomalous_exponent(self):
        return self.hurst_exponent * 2

    """
    This simulation comes from paper:

    Granik N, Weiss LE, Nehme E, Levin M, Chein M, Perlson E, Roichman Y, Shechtman Y. 
    Single-Particle Diffusion Characterization by Deep Learning. 
    Biophys J. 2019 Jul 23;117(2):185-192. doi: 10.1016/j.bpj.2019.06.015. Epub 2019 Jun 22. 
    PMID: 31280841; PMCID: PMC6701009.
    
    Original code: https://github.com/AnomDiffDB/DB/blob/master/utils.py
    """
    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        # first row of circulant matrix
        r = np.zeros(trajectory_length+1)
        r[0] = 1
        idx = np.arange(1, trajectory_length+1, 1)
        r[idx] = 0.5 * ((idx + 1) ** (2 * self.hurst_exponent) - 2 * idx ** (2 * self.hurst_exponent) + (idx - 1) ** (
            2 * self.hurst_exponent))
        r = np.concatenate((r, r[np.arange(len(r)-2, 0, -1)]))

        # get eigenvalues through fourier transform
        lamda = np.real(fftpack.fft(r))/(2*trajectory_length)

        # get trajectory using fft: dimensions assumed uncoupled
        x = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*trajectory_length)
                                                         ) + 1j*np.random.normal(size=(2*trajectory_length))))
        x = trajectory_length**(-self.hurst_exponent) * \
            np.cumsum(np.real(x[:trajectory_length]))  # rescale
        x = ((trajectory_time**self.hurst_exponent)*x)  # resulting traj. in x
        y = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*trajectory_length)
                                                         ) + 1j*np.random.normal(size=(2*trajectory_length))))
        y = trajectory_length**(-self.hurst_exponent) * \
            np.cumsum(np.real(y[:trajectory_length]))  # rescale
        y = ((trajectory_time**self.hurst_exponent)*y)  # resulting traj. in y

        t = np.arange(0,trajectory_length,1)/trajectory_length
        t = t*trajectory_time  # scale for final time T

        if np.min(x) < 0:
            x = x + np.absolute(np.min(x))  # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y))  # Add offset to y

        x = x * EXPERIMENT_PIXEL_SIZE
        y = y * EXPERIMENT_PIXEL_SIZE

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': 'hurst',
            'exponent': self.hurst_exponent,
            'info': {}
        }

class FractionalBrownianMotionSuperDiffusive(FractionalBrownianMotion):
    STRING_LABEL = 'fbm_sup'

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.SUP_DIFFUSIVE_HURST_EXPONENT_RANGE

        selected_hurst_exponent = np.random.uniform(
            selected_range[0], selected_range[1])

        return cls(hurst_exponent=selected_hurst_exponent)

class FractionalBrownianMotionSubDiffusive(FractionalBrownianMotion):
    STRING_LABEL = 'fbm_sub'

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.SUB_DIFFUSIVE_HURST_EXPONENT_RANGE

        selected_hurst_exponent = np.random.uniform(
            selected_range[0], selected_range[1])

        return cls(hurst_exponent=selected_hurst_exponent)

class FractionalBrownianMotionBrownian(FractionalBrownianMotion):
    STRING_LABEL = 'fbm_brownian'

    @classmethod
    def create_random_instance(cls):
        selected_range = cls.NOT_EXACT_BROWNIAN_HURST_EXPONENT_RANGE

        selected_hurst_exponent = np.random.uniform(
            selected_range[0], selected_range[1])

        return cls(hurst_exponent=selected_hurst_exponent)
