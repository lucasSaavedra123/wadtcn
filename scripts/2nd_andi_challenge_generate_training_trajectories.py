from DataSimulation import Andi2ndDataSimulation
from CONSTANTS import *

for i in range(100):
    Andi2ndDataSimulation().simulate_phenomenological_trajectories(TRAINING_SET_SIZE_PER_EPOCH,100,None,True,f'train_{i}', enable_parallelism=False, ignore_boundary_effects=True, type_of_simulation='models_phenom')
    exit()