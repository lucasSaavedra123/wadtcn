from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskSingleLevelPredicter import WavenetTCNMultiTaskSingleLevelPredicter

from DatabaseHandler import DatabaseHandler


L = 200
DatabaseHandler.connect_over_network(None, None, 'localhost', 'anomalous_diffusion')
WavenetTCNMultiTaskSingleLevelPredicter.analyze_hyperparameters(L, L, simulator=Andi2ndDataSimulation, initial_epochs=1, steps=1)
WavenetTCNMultiTaskSingleLevelPredicter.plot_hyperparameter_search(L, L, simulator=Andi2ndDataSimulation)
DatabaseHandler.disconnect()
