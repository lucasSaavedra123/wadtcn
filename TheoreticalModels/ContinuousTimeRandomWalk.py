import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, symmetric_alpha_levy, mittag_leffler_rand, simulate_track_time
from CONSTANTS import EXPERIMENT_PIXEL_SIZE


class ContinuousTimeRandomWalk(Model):
    STRING_LABEL = 'ctrw'
    ANOMALOUS_EXPONENT_RANGE = [0.05, 0.95]

    @classmethod
    def create_random_instance(cls):
        anomalous_exponent = np.random.uniform(
            low=cls.ANOMALOUS_EXPONENT_RANGE[0], high=cls.ANOMALOUS_EXPONENT_RANGE[1])
        model = cls(anomalous_exponent=anomalous_exponent)
        return model

    def __init__(self, anomalous_exponent=0.5, gamma=1, betha=0.5):
        self.GAMMA = gamma
        self.BETHA = betha
        self.anomalous_exponent = anomalous_exponent

    """
    This simulation comes from paper:

    Granik N, Weiss LE, Nehme E, Levin M, Chein M, Perlson E, Roichman Y, Shechtman Y. 
    Single-Particle Diffusion Characterization by Deep Learning. 
    Biophys J. 2019 Jul 23;117(2):185-192. doi: 10.1016/j.bpj.2019.06.015. Epub 2019 Jun 22. 
    PMID: 31280841; PMCID: PMC6701009.
    
    Original code: https://github.com/AnomDiffDB/DB/blob/master/utils.py
    """
    def custom_simulate_rawly(self, trajectory_length, trajectory_time):

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        jumpsX = mittag_leffler_rand(self.BETHA, trajectory_length, self.GAMMA)
        rawTimeX = np.cumsum(jumpsX)
        tX = rawTimeX*(trajectory_time)/np.max(rawTimeX)
        tX = np.reshape(tX, [len(tX), 1])

        jumpsY = mittag_leffler_rand(self.BETHA, trajectory_length, self.GAMMA)
        rawTimeY = np.cumsum(jumpsY)
        tY = rawTimeY*(trajectory_time)/np.max(rawTimeY)
        tY = np.reshape(tY, [len(tY), 1])

        x = symmetric_alpha_levy(alpha=2, n=trajectory_length, gamma=self.GAMMA ** (self.anomalous_exponent / 2))
        x = np.cumsum(x)
        x = np.reshape(x, [len(x), 1])

        y = symmetric_alpha_levy(
            alpha=2, n=trajectory_length, gamma=self.GAMMA ** (self.anomalous_exponent / 2))
        y = np.cumsum(y)
        y = np.reshape(y, [len(y), 1])

        tOut = simulate_track_time(trajectory_length, trajectory_time)
        xOut = np.zeros([trajectory_length, 1])
        yOut = np.zeros([trajectory_length, 1])
        for i in range(trajectory_length):
            xOut[i, 0] = x[find_nearest(tX, tOut[i]), 0]
            yOut[i, 0] = y[find_nearest(tY, tOut[i]), 0]

        x = np.reshape(xOut, (trajectory_length))
        y = np.reshape(yOut, (trajectory_length))
        t = tOut

        if np.min(x) < 0:
            x = x + np.absolute(np.min(x))  # Add offset to x
        if np.min(y) < 0:
            y = y + np.absolute(np.min(y))  # Add offset to y

        x = x * EXPERIMENT_PIXEL_SIZE / 10
        y = y * EXPERIMENT_PIXEL_SIZE / 10

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)

        return {
            'x': np.reshape(x, (trajectory_length)),
            'y': np.reshape(y, (trajectory_length)),
            't': t,
            'x_noisy': np.reshape(x_noisy, (trajectory_length)),
            'y_noisy': np.reshape(y_noisy, (trajectory_length)),
            'exponent_type': 'anomalous',
            'exponent': self.anomalous_exponent,
            'info': {}
        }
