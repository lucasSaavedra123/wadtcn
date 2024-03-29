import numpy as np

from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time, simulate_minflux_track_time

from CONSTANTS import EXPERIMENT_WIDTH, EXPERIMENT_HEIGHT, SIMULATE_FOR_MINFLUX
import matplotlib.pyplot as plt

class TrappingDiffusion(Model):
    STRING_LABEL = 'td'
    D_RANGE = [0.001, 1] #um2/s
    P_UNTRAP_RANGE = [0.00005, 0.00005]
    P_TRAP_RANGE = [0.00005, 0.00005]

    @classmethod
    def create_random_instance(cls):
        p_untrap = np.random.uniform(low=cls.P_UNTRAP_RANGE[0], high=cls.P_UNTRAP_RANGE[1])
        p_trap = np.random.uniform(low=cls.P_TRAP_RANGE[0], high=cls.P_TRAP_RANGE[1])
        d = np.random.choice(np.logspace(np.log10(cls.D_RANGE[0]), np.log10(cls.D_RANGE[1]), 1000))
        return cls(d, p_untrap, p_trap)

    def __init__(self, d, p_untrap, p_trap):
        assert d > 0
        assert 0 <= p_untrap <= 1
        assert 0 <= p_trap <= 1

        self.d = d * 1000000 #um2/s -> nm2/s
        self.p_untrap = p_untrap
        self.p_trap = p_trap

    def simulate_transition_array(self, length):
        states = [np.random.choice(['TRAP', 'UNTRAP'])]

        while len(states) != length:
            if states[-1] == 'TRAP':
                states.append(np.random.choice(['UNTRAP', 'TRAP'], p=[self.p_untrap, 1-self.p_untrap]))
            elif states[-1] == 'UNTRAP':
                states.append(np.random.choice(['TRAP', 'UNTRAP'], p=[self.p_trap, 1-self.p_trap]))

        states = np.vectorize({'TRAP':0,'UNTRAP':1}.get)(states)

        return states

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        """
        cells_centroids = self.__get_voronoi_centroids()
        
        initial_position = cells_centroids[np.random.randint(cells_centroids.shape[0])]
        while np.inf in initial_position:
            initial_position = cells_centroids[np.random.randint(cells_centroids.shape[0])]

        x, y = [initial_position[0]], [initial_position[1]]
        """
        if not SIMULATE_FOR_MINFLUX:
            t = simulate_track_time(trajectory_length, trajectory_time)
        else:
            t = simulate_minflux_track_time(trajectory_length, trajectory_time)

        states = self.simulate_transition_array(trajectory_length)
        displacements_x =  np.random.normal(loc=0, scale=1, size=trajectory_length) * np.sqrt(2 * self.d * t) * states
        displacements_y =  np.random.normal(loc=0, scale=1, size=trajectory_length) * np.sqrt(2 * self.d * t) * states

        x, y = np.cumsum(displacements_x), np.cumsum(displacements_y)

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, np.array(x), np.array(y), disable_offset=False)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': None,
            'exponent': None,
            'info': {
                'switching': len(np.unique(states)) == 2,
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

        plt.show()
