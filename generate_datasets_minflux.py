from DataSimulation import DeepSPTDataSimulation
from CONSTANTS import *

assert FOR_MINFLUX

for i in range(3):
   print("Train dataset", i)
   DeepSPTDataSimulation().simulate_segmentated_trajectories(TRAINING_SET_SIZE_PER_EPOCH,1_000,None,True,f'train_{i}', True)

for i in range(1):
   print("Val dataset", i)
   DeepSPTDataSimulation().simulate_segmentated_trajectories(VALIDATION_SET_SIZE_PER_EPOCH,1_000,None,True,f'val_{i}', True)
