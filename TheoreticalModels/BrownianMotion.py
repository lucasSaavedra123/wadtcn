import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time

from Trajectory import Trajectory


class BrownianMotion(Model):
    STRING_LABEL="bm"

    D_RANGE = [0.001, 1]

    def __init__(self, diffusion_coefficient):
        self.diffusion_coefficient = diffusion_coefficient

    @classmethod
    def create_random_instance(cls):
        diffusion_coefficient = np.random.uniform(low=cls.D_RANGE[0], high=cls.D_RANGE[1])
        return cls(diffusion_coefficient=diffusion_coefficient)

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        x = np.random.normal(loc=0, scale=1, size=trajectory_length)
        y = np.random.normal(loc=0, scale=1, size=trajectory_length)
        t = np.zeros(trajectory_length)

        for i in range(trajectory_length):
            if trajectory_time is None:
                delta_t = np.random.uniform(0.01e-3, 1e-3)
            else:
                delta_t = trajectory_time / trajectory_length

            x[i] = x[i] * np.sqrt(2 * self.diffusion_coefficient * delta_t)
            y[i] = y[i] * np.sqrt(2 * self.diffusion_coefficient * delta_t)
            t[i] = delta_t


        x = np.cumsum(x)
        y = np.cumsum(y)
        t = np.cumsum(t)

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': 'anomalous',
            'exponent': 1,
            'info': {
                'diffusion_coefficient': self.diffusion_coefficient
            }
        }

    def normalize_d_coefficient_to_net(self):
        delta_d = self.d_high - self.d_low
        return (1 / delta_d) * (self.diffusion_coefficient - self.d_low)

    @classmethod
    def denormalize_d_coefficient_to_net(cls, output_coefficient_net):
        delta_d = cls.d_high - cls.d_low
        return output_coefficient_net * delta_d + cls.d_low

    def get_d_coefficient(self):
        return self.diffusion_coefficient
