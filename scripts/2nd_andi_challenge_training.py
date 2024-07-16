import json

from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter

for network_class in [
    WavenetTCNMultiTaskClassifierSingleLevelPredicter,
    WavenetTCNSingleLevelAlphaPredicter,
    WavenetTCNSingleLevelDiffusionCoefficientPredicter,
]:
    network = network_class(200, None, simulator=Andi2ndDataSimulation)
    network.enable_early_stopping()
    network.fit()
    network.save_as_file()

    with open(f"./networks/{str(network)}.json", "w") as outfile:
        json.dump(network.history_training_info, outfile)
    #network.load_as_file()
    #network.trajectory_length = 100
    #network.plot_single_level_prediction(sigma=0.12)
    #network.plot_confusion_matrix(sigma=0.12)
