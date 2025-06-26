from DataSimulation import CustomDataSimulation
from PredictiveModel.RunAndTurnSegmentator import RunAndTurnSegmentator
import json


network = RunAndTurnSegmentator(1000,1000,simulator=CustomDataSimulation)
network.enable_early_stopping()
network.fit()
network.save_as_file('run_and_turn_minflux.weights.h5')

with open("networks/run_and_turn_minflux.json", "w") as info_file:
    json.dump(network.history_training_info, info_file)