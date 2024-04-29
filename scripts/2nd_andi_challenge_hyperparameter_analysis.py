from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNModelSingleLevelPredicter import WavenetTCNModelSingleLevelPredicter

from DatabaseHandler import DatabaseHandler


L = 200

DatabaseHandler.connect_over_network(None, None, 'localhost', 'anomalous_diffusion')
WavenetTCNModelSingleLevelPredicter.analyze_hyperparameters(L, None, simulator=Andi2ndDataSimulation)
WavenetTCNModelSingleLevelPredicter.plot_hyperparameter_search(L, None, simulator=Andi2ndDataSimulation)
DatabaseHandler.disconnect()
