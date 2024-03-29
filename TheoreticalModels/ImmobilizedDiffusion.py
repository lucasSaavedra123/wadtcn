import numpy as np

from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time, simulate_minflux_track_time

from CONSTANTS import EXPERIMENT_WIDTH, EXPERIMENT_HEIGHT, SIMULATE_FOR_MINFLUX
import matplotlib.pyplot as plt

class TrappingDiffusion(Model):
    STRING_LABEL = 'i'

    @classmethod
    def create_random_instance(cls):
        return cls()

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        x = np.random.uniform(0, EXPERIMENT_WIDTH) * np.ones(trajectory_length)
        y = np.random.uniform(0, EXPERIMENT_HEIGHT) * np.ones(trajectory_length)

        if not SIMULATE_FOR_MINFLUX:
            t = simulate_track_time(trajectory_length, trajectory_time)
        else:
            t = simulate_minflux_track_time(trajectory_length, trajectory_time)

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, np.array(x), np.array(y), disable_offset=True)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': None,
            'exponent': None,
            'info': {}
        }

    def plot(self, trajectory, with_noise=False):
        plt.suptitle(r"$P_{untrap}="+str(np.round(self.p_untrap, 2))+r"$, $P_{untrap}="+str(np.round(self.p_trap, 2))+r"$, $D="+str(np.round(self.d/1000000, 3))+r"\mu m^{2}/s$")
        plt.plot(trajectory.get_x(), trajectory.get_y(), marker="X", color='black')
        if with_noise:
            plt.plot(trajectory.get_noisy_x(), trajectory.get_noisy_y(), marker="X", color='red')

        plt.show()
