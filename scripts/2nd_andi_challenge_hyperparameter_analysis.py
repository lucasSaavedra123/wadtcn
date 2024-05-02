from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskSingleLevelPredicter import WavenetTCNMultiTaskSingleLevelPredicter

from DatabaseHandler import DatabaseHandler


L = 200
DatabaseHandler.connect_over_network(None, None, 'localhost', 'anomalous_diffusion')
WavenetTCNMultiTaskSingleLevelPredicter.analyze_hyperparameters(L, None, simulator=Andi2ndDataSimulation)
WavenetTCNMultiTaskSingleLevelPredicter.plot_hyperparameter_search(L, None, simulator=Andi2ndDataSimulation)
DatabaseHandler.disconnect()
