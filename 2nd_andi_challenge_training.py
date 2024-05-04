from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskSingleLevelPredicter import WavenetTCNMultiTaskSingleLevelPredicter


L = 200
network = WavenetTCNMultiTaskSingleLevelPredicter(L, L, simulator=Andi2ndDataSimulation)
network.fit()
network.save_as_file()
