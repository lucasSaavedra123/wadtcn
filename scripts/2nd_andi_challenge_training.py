import json

from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter

for network_class in [
    WavenetTCNSingleLevelAlphaPredicter,
    WavenetTCNSingleLevelDiffusionCoefficientPredicter,
    WavenetTCNMultiTaskClassifierSingleLevelPredicter,
]:
    network = network_class(100, None, simulator=Andi2ndDataSimulation)
    network.enable_early_stopping()
    network.fit()
    network.save_as_file()

    with open(f"./networks/{str(network)}.json", "w") as outfile:
        json.dump(network.history_training_info, outfile)
    #network.load_as_file()
    #network.plot_single_level_prediction()
    #network.plot_confusion_matrix()