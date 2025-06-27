from DataSimulation import CustomDataSimulation
from PredictiveModel.RunAndTurnSegmentator import RunAndTurnSegmentator
import matplotlib.pyplot as plt
import json
import tqdm

TRAIN = True

network = RunAndTurnSegmentator(1000,1000,simulator=CustomDataSimulation)

if TRAIN:
    network.enable_early_stopping()
    network.fit()
    network.save_as_file('run_and_turn_minflux.weights.h5')

    with open("networks/run_and_turn_minflux.json", "w") as info_file:
        json.dump(network.history_training_info, info_file)
else:
    network.load_as_file('run_and_turn_minflux.weights.h5')

lengths = list(range(25,1000,25))
scores = []
for length in tqdm.tqdm(lengths):
    network.trajectory_length = length
    scores.append(network.f1_score())

plt.plot(lengths,scores)
plt.show()

network.plot_confusion_matrix()
