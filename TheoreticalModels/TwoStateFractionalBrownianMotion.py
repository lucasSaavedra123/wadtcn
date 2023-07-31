import numpy as np

from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset

def simulate_track_time(track_length, track_time):
    t = np.linspace(0, track_time, track_length)
    return t

class TwoStateFractionalBrownianMotion(Model):
    """
    State-0: Free Diffusion
    State-1: Subdiffusive FBM State
    """
    STRING_LABEL = 'tfbm'
    D_RANGE = [0.05, 0.2]
    K0_RANGE = [0.01, 0.08]
    K1_RANGE = [0.007, 0.2]
    H1_RANGE = [0.05, 0.45]

    @classmethod
    def create_random_instance(cls):
        # k_state(i) dimensions = 1 / frame
        # D_state(i) dimensions = um^2 * s^(-beta)
        d0_state = np.random.uniform(low=cls.D_RANGE[0], high=cls.D_RANGE[1])
        d1_state = np.random.uniform(low=cls.D_RANGE[0], high=cls.D_RANGE[1])
        k_state0 = np.random.uniform(low=cls.K0_RANGE[0], high=cls.K0_RANGE[1])
        k_state1 = np.random.uniform(low=cls.K1_RANGE[0], high=cls.K1_RANGE[1])
        h_state1 = np.random.uniform(low=cls.H1_RANGE[0], high=cls.H1_RANGE[1])
        return cls(k_state0, k_state1, d0_state, d1_state, h_state1)

    def __init__(self, k_state0, k_state1, d0_state, d1_state, h_state1):
        assert (d0_state > 0), "Invalid Diffusion coefficient state-0"
        assert (d1_state > 0), "Invalid Diffusion coefficient state-1"
        assert (k_state0 > 0), "Invalid switching rate state-0"
        assert (k_state1 > 0), "Invalid switching rate state-1"
        assert (h_state1 > 0) and (h_state1 < 0.5), "Invalid hurst exponent state-1"
        self.k_state0 = k_state0
        self.k_state1 = k_state1
        self.h_state1 = h_state1
        self.d_state0 = d0_state * 1e6  # Convert from um^2 -> nm^2
        self.d_state1 = d1_state * 1e6  # Convert from um^2 -> nm^2

    """
    def get_d_state0(self):
        return self.D_state0 / 1000000

    def normalize_d_coefficient_to_net(self, state_number):
        assert (state_number == 0), "Not a valid state"
        delta_d0 = self.d0_high - self.d0_low
        return (1 / delta_d0) * (self.get_d_state0() - self.d0_low)

    @classmethod
    def denormalize_d_coefficient_to_net(cls, output_coefficient_net):
        delta_d0 = cls.d0_high - cls.d0_low
        return output_coefficient_net * delta_d0 + cls.d0_low
    """

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        x = np.zeros(shape=trajectory_length)
        y = np.zeros(shape=trajectory_length)

        state, switching = self.simulate_switching_states(trajectory_length)

        SIMULATION_NUMBER_OF_SUBSTEPS = 50
        I = SIMULATION_NUMBER_OF_SUBSTEPS * trajectory_length
        t = np.arange(0, trajectory_time, trajectory_time/trajectory_length)

        M1 = 0
        M2 = trajectory_time

        assert M2 > M1, f"M1={M1}, M2={M2}"

        S = np.linspace(M1, M2, I)
        LAMBDA = (M2-M1)/I

        X_NORMAL_VALUES = np.random.normal(0,LAMBDA,size=I)
        Y_NORMAL_VALUES = np.random.normal(0,LAMBDA,size=I)

        H = np.zeros(shape=I)
        D = np.zeros(shape=I)

        h_dict = {0:0.5, 1:self.h_state1}
        d_dict = {0:self.d_state0/1e6, 1:self.d_state1/1e6}

        time_H = np.vectorize(h_dict.get)(state)
        time_D = np.vectorize(d_dict.get)(state)

        for i in range(trajectory_length):
            H[i*SIMULATION_NUMBER_OF_SUBSTEPS:(i+1)*SIMULATION_NUMBER_OF_SUBSTEPS] = time_H[i]
            D[i*SIMULATION_NUMBER_OF_SUBSTEPS:(i+1)*SIMULATION_NUMBER_OF_SUBSTEPS] = time_D[i]

        for index, t_i in enumerate(t):
            accumulate_value_x = 0
            accumulate_value_y = 0

            for value in range(index*SIMULATION_NUMBER_OF_SUBSTEPS):
                accumulate_value_x += X_NORMAL_VALUES[value] * (np.sqrt(2*D[value]*H[value]) * ((t_i-S[value]) ** (H[value]-(1/2))))
                accumulate_value_y += Y_NORMAL_VALUES[value] * (np.sqrt(2*D[value]*H[value]) * ((t_i-S[value]) ** (H[value]-(1/2))))

            x[index] = accumulate_value_x
            y[index] = accumulate_value_y

        x = x * 1000
        y = y * 1000

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, x, y)
        #t = simulate_track_time(trajectory_length, trajectory_time)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': None,
            'exponent': None,
            'info': {'switching': switching, 'state': state, 'h': time_H, 'd': time_D, 'h_0': 0.5, 'h_1': self.h_state1, 'd_0': self.d_state0, 'd_1': self.d_state1}
        }

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
