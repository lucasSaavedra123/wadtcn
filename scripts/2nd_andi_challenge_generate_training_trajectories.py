from DataSimulation import Andi2ndDataSimulation
from CONSTANTS import *

for i in range(3):
   print("Train Regression dataset", i)
   Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_regression_training(TRAINING_SET_SIZE_PER_EPOCH,200,None,True,f'train_{i}', ignore_boundary_effects=True)

for i in range(2):
   print("Val Regression dataset", i)
   Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_regression_training(VALIDATION_SET_SIZE_PER_EPOCH,200,None,True,f'val_{i}', ignore_boundary_effects=True)

for i in range(3):
   print("Train Classification dataset", i)
   Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_classification_training(TRAINING_SET_SIZE_PER_EPOCH,200,None,True,f'train_{i}', ignore_boundary_effects=True, enable_parallelism=True, type_of_simulation='models_phenom')

for i in range(2):
   print("Val Regression dataset", i)
   Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_classification_training(VALIDATION_SET_SIZE_PER_EPOCH,200,None,True,f'val_{i}', ignore_boundary_effects=True, enable_parallelism=True, type_of_simulation='models_phenom')
