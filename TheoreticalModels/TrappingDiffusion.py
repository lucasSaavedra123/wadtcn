import numpy as np

from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time

from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi, voronoi_plot_2d
from CONSTANTS import EXPERIMENT_WIDTH, EXPERIMENT_HEIGHT
import matplotlib.pyplot as plt

class TrappingDiffusion(Model):
    STRING_LABEL = 'td'
    D_RANGE = [0.001, 1] #um2/s
    P_UNTRAP_RANGE = [0.0001, 0.0005]
    P_TRAP_RANGE = [0.0001, 0.0005]

    @classmethod
    def create_random_instance(cls):
        p_untrap = np.random.uniform(low=cls.P_UNTRAP_RANGE[0], high=cls.P_UNTRAP_RANGE[1])
        p_trap = np.random.uniform(low=cls.P_TRAP_RANGE[0], high=cls.P_TRAP_RANGE[1])
        d = np.random.choice(np.logspace(np.log10(cls.D_RANGE[0]), np.log10(cls.D_RANGE[1]), 1000))
        return cls(d, p_untrap, p_trap)

    def __init__(self, d, p_untrap, p_trap):
        assert d > 0
        assert 0 < p_untrap < 1
        assert 0 < p_trap < 1

        self.d = d * 1000000 #um2/s -> nm2/s
        self.p_untrap = p_untrap
        self.p_trap = p_trap

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        """
        cells_centroids = self.__get_voronoi_centroids()
        
        initial_position = cells_centroids[np.random.randint(cells_centroids.shape[0])]
        while np.inf in initial_position:
            initial_position = cells_centroids[np.random.randint(cells_centroids.shape[0])]

        x, y = [initial_position[0]], [initial_position[1]]
        """
        x, y = [np.random.uniform(0, EXPERIMENT_WIDTH)], [np.random.uniform(0, EXPERIMENT_HEIGHT)]
        current_state = np.random.choice(['TRAP', 'UNTRAP'])
        switching = False

        while len(x) != trajectory_length:
            if current_state == 'TRAP':
                x.append(x[-1])
                y.append(y[-1])
            elif current_state == 'UNTRAP':
                x.append(x[-1] + np.random.normal(loc=0, scale=1) * np.sqrt(2 * self.d * (trajectory_time/trajectory_length)))
                y.append(y[-1] + np.random.normal(loc=0, scale=1) * np.sqrt(2 * self.d * (trajectory_time/trajectory_length)))

            if current_state == 'TRAP':
                current_state = np.random.choice(['TRAP', 'UNTRAP'], p=[self.p_untrap, 1-self.p_untrap])
                switching = True
            elif current_state == 'UNTRAP':
                current_state = np.random.choice(['TRAP', 'UNTRAP'], p=[1-self.p_trap, self.p_trap])
                switching = True

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, np.array(x), np.array(y), disable_offset=False)
        t = simulate_track_time(trajectory_length, trajectory_time)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': None,
            'exponent': None,
            'info': {
                'switching': switching,
                'p_untrap': self.p_untrap,
                'p_trap': self.p_trap,
                'd': self.d,
            }
        }

    def plot(self, trajectory, with_noise=False):
        plt.suptitle(r"$P_{untrap}="+str(np.round(self.p_untrap, 2))+r"$, $P_{untrap}="+str(np.round(self.p_trap, 2))+r"$, $D="+str(np.round(self.d/1000000, 3))+r"\mu m^{2}/s$")
        plt.plot(trajectory.get_x(), trajectory.get_y(), marker="X", color='black')
        if with_noise:
            plt.plot(trajectory.get_noisy_x(), trajectory.get_noisy_y(), marker="X", color='red')

        plt.xlim([np.min(trajectory.get_x()) * 0.95, np.max(trajectory.get_x()) * 1.05])
        plt.ylim([np.min(trajectory.get_y()) * 0.95, np.max(trajectory.get_y()) * 1.05])

        plt.show()
