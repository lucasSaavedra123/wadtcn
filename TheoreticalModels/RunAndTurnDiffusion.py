import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import IntermittentSelfPropelledParticle

class RunAndTurnDiffusion(Model):
    STRING_LABEL="run_and_turn"

    def create_particle(self):
        psi_r = lambda: np.random.exponential(1.0)
        psi_t = lambda: np.random.exponential(0.5)
        p_chi = lambda: np.random.uniform(-np.pi, np.pi)

        return IntermittentSelfPropelledParticle(
            v0=1.0,
            Dr_phi=0.1,
            Dt_phi=1.0,
            D=np.random.uniform(0.001, 2),
            psi_r_sampler=psi_r,
            psi_t_sampler=psi_t,
            p_chi_sampler=p_chi,
            dt=np.random.uniform(0.0001, 0.0100)
        )

    @classmethod
    def create_random_instance(cls):
        return cls()

    def __init__(self):
        pass

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        particle = self.create_particle()
        trajectory, states = particle.simulate(steps=trajectory_length, return_states=True)
        x = trajectory[:,0]
        y = trajectory[:,1]

        noise_x = np.random.normal(0.007, 0.001, size=x.shape)*np.random.choice([-1,1], size=x.shape)
        noise_y = np.random.normal(0.007, 0.001, size=y.shape)*np.random.choice([-1,1], size=y.shape)

        noisy_x = x + noise_x
        noisy_y = y + noise_y

        t = np.arange(trajectory_length) * particle.dt

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': noisy_x,
            'y_noisy': noisy_y,
            'exponent_type': 'anomalous',
            'exponent': 1,
            'info': {
                'state': states,
            }
        }
