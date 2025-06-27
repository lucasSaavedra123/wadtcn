from DataSimulation import CustomDataSimulation
from PredictiveModel.RunAndTurnSegmentator import RunAndTurnSegmentator
import json

TRAIN = True

network = RunAndTurnSegmentator(200,200,simulator=CustomDataSimulation)

if TRAIN:
    network.enable_early_stopping()
    network.fit()
    network.save_as_file('run_and_turn_minflux.weights.h5')

    with open("networks/run_and_turn_minflux.json", "w") as info_file:
        json.dump(network.history_training_info, info_file)
else:
    network.load_as_file('run_and_turn_minflux.weights.h5')
    network.plot_confusion_matrix()