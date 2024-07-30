from DataSimulation import AndiDataSimulation
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter

"""
net = WavenetTCNSingleLevelAlphaPredicter(200,200,simulator=AndiDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()
"""
"""
net = WavenetTCNMultiTaskClassifierSingleLevelPredicter(200,200,simulator=AndiDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()
"""
"""
net = WavenetTCNSingleLevelChangePointPredicter(200,200,simulator=AndiDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()
"""