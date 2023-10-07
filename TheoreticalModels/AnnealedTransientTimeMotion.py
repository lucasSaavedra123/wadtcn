import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, generate_diffusion_coefficient_and_transit_time_pair, simulate_track_time


class AnnealedTransientTimeMotion(Model):
    STRING_LABEL = 'attm'
    ANOMALOUS_EXPONENT_RANGE = [0.05, 0.95]
    REGIMES = [1]
    DIFFUSION_COEFFICIENT_RANGE = [0.01, 0.5]

    @classmethod
    def create_random_instance(cls):
        selected_regime = np.random.choice(cls.REGIMES)
        if selected_regime == 0:
            anomalous_exponent = 1
        else:
            anomalous_exponent = np.random.uniform(
                low=cls.ANOMALOUS_EXPONENT_RANGE[0], high=cls.ANOMALOUS_EXPONENT_RANGE[1])

        return cls(anomalous_exponent=anomalous_exponent, regime=selected_regime)

    def __init__(self, anomalous_exponent=0.5, regime=1):
        if regime not in [0, 1, 2]:
            raise ValueError('ATTM has only three regimes: 0, 1 or 2.')
        if anomalous_exponent > 1:
            raise ValueError('ATTM only allows for anomalous exponents <= 1.')

        self.regime = regime
        self.anomalous_exponent = anomalous_exponent

    """
    This simulation comes from paper:

    Carlo Manzo, Juan A. Torreno-Pina, Pietro Massignan, Gerald J. Lapeyre, Jr.,
    Maciej Lewenstein, and Maria F. Garcia Parajo

    Weak Ergodicity Breaking of Receptor Motion in Living Cells Stemming 
    from Random Diffusivity

    Original code: Not Available

    And paper:

    Muñoz-Gil, G., Volpe, G., Garcia-March, M.A. et al.
    Objective comparison of methods to decode anomalous diffusion.
    Nat Commun 12, 6253 (2021). https://doi.org/10.1038/s41467-021-26320-w

    Original Code: https://github.com/AnDiChallenge/andi_datasets/blob/master/functions/diffusion_models.py
    """
    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        # Gamma and sigma selection
        if self.regime == 0:
            sigma = 3*np.random.rand()
            gamma = np.random.uniform(low = -5, high = sigma)
            if self.anomalous_exponent < 1:
                raise ValueError('ATTM regime 0 only allows for anomalous exponents = 1.')  
        elif self.regime == 1:
            sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
            gamma = sigma/self.anomalous_exponent
            while sigma > gamma or gamma > sigma + 1:
                sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
                gamma = sigma/self.anomalous_exponent
        elif self.regime == 2:
            gamma = 1/(1-self.anomalous_exponent)
            sigma = np.random.uniform(low = 1e-2, high = gamma-1)
        # Generate the trajectory  
        posX = np.array([0])
        posY = np.array([0])

        time_per_step = trajectory_time/trajectory_length

        extensions = 3

        ds = []

        while len(posX) < extensions * trajectory_length:
            d, t = generate_diffusion_coefficient_and_transit_time_pair(sigma, gamma, 0.12, 0.10)

            while 0.05 < d:
                d, t = generate_diffusion_coefficient_and_transit_time_pair(sigma, gamma, 0.12, 0.10)

            number_of_steps =int(t/time_per_step)

            if trajectory_length < number_of_steps:
                number_of_steps = trajectory_length

            ds += [d] * number_of_steps

            distX = np.sqrt(2*d*time_per_step)*np.random.randn(number_of_steps)
            distY = np.sqrt(2*d*time_per_step)*np.random.randn(number_of_steps)                
            posX = np.append(posX, distX)
            posY = np.append(posY, distY)

        cut_index = np.random.randint(0, len(posX) - trajectory_length)
        posX, posY = np.cumsum(posX)[cut_index:trajectory_length+cut_index] * 1000, np.cumsum(posY)[cut_index:trajectory_length+cut_index] * 1000 #*1000 to nm
        ds = ds[cut_index:trajectory_length+cut_index]

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, posX, posY)
        t = simulate_track_time(trajectory_length, trajectory_time)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': 'anomalous',
            'exponent': self.anomalous_exponent,
            'info': {'ds':ds}
        }
