from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskSingleLevelPredicter import WavenetTCNMultiTaskSingleLevelPredicter


L = 100
network = WavenetTCNMultiTaskSingleLevelPredicter(L, None, simulator=Andi2ndDataSimulation)

try:
    network.load_as_file()
except:
    network.enable_early_stopping()
    network.fit()
    network.save_as_file()

network.plot_confusion_matrix()
network.plot_single_level_prediction()