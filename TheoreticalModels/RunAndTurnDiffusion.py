import numpy as np
from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import IntermittentSelfPropelledParticle

class RunAndTurnDiffusion(Model):
    STRING_LABEL="run_and_turn"

    def create_particle(self):
        return IntermittentSelfPropelledParticle(
            v0=np.random.uniform(1,0.001),          # Velocidad de propulsión
            D=np.random.uniform(1,0.001),          # Difusión térmica
            D_phi_run=np.random.uniform(0,0.1),#0.0001,   # Difusión rotacional durante "run"
            D_phi_turn=np.random.uniform(0.5,2),  # Difusión rotacional durante "turn"
            p_flip=0.5,      # Probabilidad de flip angular
            dt=1          # Paso de tiempo
        )

    @classmethod
    def create_random_instance(cls):
        return cls()

    def __init__(self):
        pass

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        particle = self.create_particle()
        particle.evolve(steps=trajectory_length, psi_r=0.001, psi_t=0.001)
        trajectory, angles, states = particle.get_trajectory()

        states = [1 if s=='turn' else 0 for s in states]

        x = trajectory[:,0]/200
        y = trajectory[:,1]/200

        noise_x = np.random.normal(0.007, 0.001, size=x.shape)*np.random.choice([-1,1], size=x.shape) * 2.0
        noise_y = np.random.normal(0.007, 0.001, size=y.shape)*np.random.choice([-1,1], size=y.shape) * 2.0

        noisy_x = x + noise_x
        noisy_y = y + noise_y

        t = np.arange(trajectory_length) * particle.dt

        return {
            'x': x[:trajectory_length],
            'y': y[:trajectory_length],
            't': t[:trajectory_length],
            'x_noisy': noisy_x[:trajectory_length],
            'y_noisy': noisy_y[:trajectory_length],
            'exponent_type': 'anomalous',
            'exponent': 1,
            'info': {
                'state': states[:trajectory_length],
            }
        }
