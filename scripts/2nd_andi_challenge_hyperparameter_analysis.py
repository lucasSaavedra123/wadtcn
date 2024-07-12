from DatabaseHandler import DatabaseHandler
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter

#DatabaseHandler.connect_to_local('anomalous_diffusion')
DatabaseHandler.connect_over_network(None, None, '192.168.0.173', 'anomalous_diffusion')

"""
SET IN CONSTANTS TRAINING SET SIZE OF 100_000
"""

for network_class in [
    WavenetTCNSingleLevelAlphaPredicter,
    WavenetTCNSingleLevelDiffusionCoefficientPredicter,
    WavenetTCNMultiTaskClassifierSingleLevelPredicter,
    WavenetTCNSingleLevelChangePointPredicter,
]:
    network_class.analyze_hyperparameters(200, None, initial_epochs=1, steps=1, simulator=Andi2ndDataSimulation)

DatabaseHandler.disconnect()
