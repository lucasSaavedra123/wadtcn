from DataSimulation import DeepSPTDataSimulation
from CONSTANTS import *

for i in range(3):
   print("Train Regression dataset", i)
   DeepSPTDataSimulation().simulate_segmentated_trajectories(TRAINING_SET_SIZE_PER_EPOCH,1_000,None,True,f'train_{i}')

for i in range(1):
   print("Val Regression dataset", i)
   DeepSPTDataSimulation().simulate_segmentated_trajectories(VALIDATION_SET_SIZE_PER_EPOCH,1_000,None,True,f'val_{i}')
