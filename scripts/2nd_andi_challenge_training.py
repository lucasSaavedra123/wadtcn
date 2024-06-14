from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
#from PredictiveModel.WavenetTCNMultiTaskSingleLevelPredicter import WavenetTCNMultiTaskSingleLevelPredicter


network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(100, None, simulator=Andi2ndDataSimulation)
#network = WavenetTCNMultiTaskSingleLevelPredicter(100, None, simulator=Andi2ndDataSimulation)

network.enable_early_stopping()
network.fit()

network.save_as_file()
network.load_as_file()
network.plot_confusion_matrix()
#network.plot_single_level_prediction()