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

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__