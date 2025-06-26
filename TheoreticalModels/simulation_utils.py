import numpy as np
import sys, os

"""
This method comes from paper:

Granik N, Weiss LE, Nehme E, Levin M, Chein M, Perlson E, Roichman Y, Shechtman Y. 
Single-Particle Diffusion Characterization by Deep Learning. 
Biophys J. 2019 Jul 23;117(2):185-192. doi: 10.1016/j.bpj.2019.06.015. Epub 2019 Jun 22. 
PMID: 31280841; PMCID: PMC6701009.

Original code: https://github.com/AnomDiffDB/DB/blob/master/utils.py
"""

def mittag_leffler_rand(beta = 0.5, n = 1000, gamma = 1):
    t = -np.log(np.random.uniform(size=[n,1]))
    u = np.random.uniform(size=[n,1])
    w = np.sin(beta*np.pi)/np.tan(beta*np.pi*u)-np.cos(beta*np.pi)
    t = t*((w**1/(beta)))
    t = gamma*t

    return t

"""
This method comes from paper:

Granik N, Weiss LE, Nehme E, Levin M, Chein M, Perlson E, Roichman Y, Shechtman Y. 
Single-Particle Diffusion Characterization by Deep Learning. 
Biophys J. 2019 Jul 23;117(2):185-192. doi: 10.1016/j.bpj.2019.06.015. Epub 2019 Jun 22. 
PMID: 31280841; PMCID: PMC6701009.

Original code: https://github.com/AnomDiffDB/DB/blob/master/utils.py
"""

def symmetric_alpha_levy(alpha = 0.5,n=1000,gamma = 1):
    u = np.random.uniform(size=[n,1])
    v = np.random.uniform(size=[n,1])
    
    phi = np.pi*(v-0.5)
    w = np.sin(alpha*phi)/np.cos(phi)
    z = -1*np.log(u)*np.cos(phi)
    z = z/np.cos((1-alpha)*phi)
    x = gamma*w*z**(1-(1/alpha))
    
    return x

"""
This method comes from paper:

Carlo Manzo, Juan A. Torreno-Pina, Pietro Massignan, Gerald J. Lapeyre, Jr.,
Maciej Lewenstein, and Maria F. Garcia Parajo

Weak Ergodicity Breaking of Receptor Motion in Living Cells Stemming 
from Random Diffusivity

Original code: Not Available
"""

def generate_diffusion_coefficient_and_transit_time_pair(sigma, gamma, b, k):
    d = np.random.gamma(sigma, b)
    t = np.random.gamma(1, k/(d**gamma))
    return d, t

def add_custom_noise(track_length):
    # New error formula
    mean_error = 40
    sigma_error = 10
    error_x = np.random.normal(loc=mean_error / 2, scale=sigma_error / 2, size=track_length)
    error_x_sign = np.random.choice([-1, 1], size=track_length)
    error_y = np.random.normal(loc=mean_error / 2, scale=sigma_error / 2, size=track_length)
    error_y_sign = np.random.choice([-1, 1], size=track_length)
    return error_x * error_x_sign, error_y * error_y_sign

def add_noise_and_offset(track_length, x, y):
    noise_x, noise_y = add_custom_noise(track_length)
    x_noisy = x + noise_x
    y_noisy = y + noise_y
    if np.min(x_noisy) < np.min(x) and np.min(x_noisy) < 0:
        min_noisy_x = np.absolute(np.min(x_noisy))
        x_noisy = x_noisy + min_noisy_x  # Convert to positive
        x = x + min_noisy_x
    if np.min(x_noisy) > np.min(x) and np.min(x) < 0:
        min_x = np.absolute(np.min(x))
        x_noisy = x_noisy + min_x  # Convert to positive
        x = x + min_x
    if np.min(y_noisy) < np.min(y) and np.min(y_noisy) < 0:
        min_noisy_y = np.absolute(np.min(y_noisy))
        y_noisy = y_noisy + min_noisy_y  # Convert to positive
        y = y + min_noisy_y
    if np.min(y_noisy) > np.min(y) and np.min(y) < 0:
        min_y = np.absolute(np.min(y))
        y_noisy = y_noisy + min_y  # Convert to positive
        y = y + min_y
    offset_x = np.ones(shape=track_length) * np.random.uniform(low=0, high=(
            10000 - np.minimum(np.max(x), np.max(x_noisy))))
    offset_y = np.ones(shape=track_length) * np.random.uniform(low=0, high=(
            10000 - np.minimum(np.max(y), np.max(y_noisy))))
    x = x + offset_x
    y = y + offset_y
    x_noisy = x_noisy + offset_x
    y_noisy = y_noisy + offset_y
    return x, x_noisy, y, y_noisy

def simulate_track_time(track_length, track_time):
    #return np.linspace(0, track_time, track_length)
    delta = track_time / track_length
    return np.arange(0,track_length,1) * delta

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class IntermittentSelfPropelledParticle:
    def __init__(self, v0, Dr_phi, Dt_phi, D, psi_r_sampler, psi_t_sampler, p_chi_sampler, dt=0.01):
        """
        - v0: velocidad constante en fase 'run'
        - Dr_phi: coef. de difusión rotacional durante 'run'
        - Dt_phi: coef. de difusión rotacional durante 'turn'
        - D: coef. de difusión traslacional (Browniano)
        - psi_r_sampler: función que genera tiempos de 'run'
        - psi_t_sampler: función que genera tiempos de 'turn'
        - p_chi_sampler: función que genera ángulo de reorientación χ
        - dt: paso de tiempo de integración
        """
        self.v0 = v0
        self.Dr_phi = Dr_phi
        self.Dt_phi = Dt_phi
        self.D = D
        self.psi_r_sampler = psi_r_sampler
        self.psi_t_sampler = psi_t_sampler
        self.p_chi_sampler = p_chi_sampler
        self.dt = dt

        self.r = np.zeros(2)
        self.phi = 2 * np.pi * np.random.rand()
        self.mode = 'run'
        self.time_in_mode = 0
        self.current_mode_duration = self.psi_r_sampler()

    def _update_orientation(self, D_phi):
        dphi = np.sqrt(2 * D_phi * self.dt) * np.random.randn()
        self.phi = (self.phi + dphi) % (2 * np.pi)

    def step(self):
        if self.time_in_mode >= self.current_mode_duration:
            if self.mode == 'run':
                self.mode = 'turn'
                self.current_mode_duration = self.psi_t_sampler()
            else:
                self.mode = 'run'
                self.current_mode_duration = self.psi_r_sampler()
                self.phi += self.p_chi_sampler()  # flip angular
                self.phi %= 2 * np.pi  # asegura ángulo en [0, 2π]
            self.time_in_mode = 0

        if self.mode == 'run':
            direction = np.array([np.cos(self.phi), np.sin(self.phi)])
            velocity = self.v0 * direction
            D_phi = self.Dr_phi
        else:
            velocity = np.zeros(2)
            D_phi = self.Dt_phi

        noise = np.sqrt(2 * self.D * self.dt) * np.random.randn(2)
        self.r += velocity * self.dt + noise
        self._update_orientation(D_phi)

        self.time_in_mode += self.dt
        return self.r.copy(), self.mode

    def simulate(self, T=None, steps=None, return_states=False):
        """
        Simula la trayectoria de la partícula por un tiempo total T.
        """
        steps = int(T / self.dt) if T is not None else steps
        trajectory = np.zeros((steps, 2))
        states = []
        for i in range(steps):
            trajectory[i], state = self.step()
            states.append(0 if state == 'run' else 1)

        return trajectory if not return_states else (trajectory, states)
