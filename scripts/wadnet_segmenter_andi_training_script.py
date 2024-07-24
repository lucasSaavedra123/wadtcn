from DataSimulation import AndiDataSimulation
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter

net = WavenetTCNSingleLevelAlphaPredicter(200,200,simulator=AndiDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()
