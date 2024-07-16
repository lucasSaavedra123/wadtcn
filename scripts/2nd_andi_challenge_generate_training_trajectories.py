from DataSimulation import Andi2ndDataSimulation
from CONSTANTS import *

for i in range(5):
   print("Regression dataset", i)
   Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_regression_training(TRAINING_SET_SIZE_PER_EPOCH,200,None,True,f'train_{i}', ignore_boundary_effects=True)

for i in range(5):
   print("Classification dataset", i)
   Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_classification_training(TRAINING_SET_SIZE_PER_EPOCH,200,None,True,f'train_{i}', ignore_boundary_effects=True, enable_parallelism=True, type_of_simulation='models_phenom')
