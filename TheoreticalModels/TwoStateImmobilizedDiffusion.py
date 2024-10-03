import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time

from Trajectory import Trajectory


class TwoStateImmobilizedDiffusion(Model):
    STRING_LABEL="id"

    D_RANGE = [0.001, 1]
    K0_RANGE = [0.01, 0.08]
    K1_RANGE = [0.007, 0.2]

    @classmethod
    def create_random_instance(cls):
        diffusion_coefficient = np.random.uniform(low=cls.D_RANGE[0], high=cls.D_RANGE[1])
        k_state0 = np.random.uniform(low=cls.K0_RANGE[0], high=cls.K0_RANGE[1])
        k_state1 = np.random.uniform(low=cls.K1_RANGE[0], high=cls.K1_RANGE[1])
        return cls(k_state0, k_state1, diffusion_coefficient)

    def __init__(self, k_state0, k_state1, diffusion_coefficient):
        assert (diffusion_coefficient > 0), "Invalid Diffusion coefficient state-0"
        assert (k_state0 > 0), "Invalid switching rate state-0"
        assert (k_state0 > 0), "Invalid switching rate state-1"
        self.k_state0 = k_state0
        self.k_state1 = k_state1
        self.diffusion_coefficient = diffusion_coefficient * 1000000  # Convert from um^2 -> nm^2

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        x = np.random.normal(loc=0, scale=1, size=trajectory_length)
        y = np.random.normal(loc=0, scale=1, size=trajectory_length)

        state, switching = self.simulate_switching_states(trajectory_length)

        for i in range(trajectory_length):
            x[i] = x[i] * np.sqrt(2 * self.diffusion_coefficient * (trajectory_time / trajectory_length)) * (1-state[i])
            y[i] = y[i] * np.sqrt(2 * self.diffusion_coefficient * (trajectory_time / trajectory_length)) * (1-state[i])

        x = np.cumsum(x)
        y = np.cumsum(y)

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)

        t = simulate_track_time(trajectory_length, trajectory_time)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': 'anomalous',
            'exponent': 1,
            'info': {
                'state': state,
                'switching': switching,
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

    def simulate_switching_states(self, trajectory_length):
        # Residence time
        res_time0 = 1 / self.k_state0
        res_time1 = 1 / self.k_state1

        # Compute each t_state according to exponential laws
        t_state0 = np.random.exponential(scale=res_time0, size=trajectory_length)
        t_state1 = np.random.exponential(scale=res_time1, size=trajectory_length)

        # Set initial t_state for each state
        t_state0_next = 0
        t_state1_next = 0

        # Pick an initial state from a random choice
        current_state = np.random.choice([0, 1])

        # Detect real switching behavior
        switching = ((current_state == 0) and (int(np.ceil(t_state0[t_state0_next])) < trajectory_length)) or (
                (current_state == 1) and (int(np.ceil(t_state1[t_state1_next])) < trajectory_length))

        # Fill state array
        state = np.zeros(shape=trajectory_length)
        i = 0

        while i < trajectory_length:
            if current_state == 1:
                current_state_length = int(np.ceil(t_state1[t_state1_next]))

                if (current_state_length + i) < trajectory_length:
                    state[i:(i + current_state_length)] = np.ones(shape=current_state_length)
                else:
                    state[i:trajectory_length] = np.ones(shape=(trajectory_length - i))

                current_state = 0  # Set state from 1->0
            else:
                current_state_length = int(np.ceil(t_state0[t_state0_next]))
                current_state = 1  # Set state from 0->1

            i += current_state_length

        return state, switching
