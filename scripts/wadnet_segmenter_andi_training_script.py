from DataSimulation import AndiDataSimulation, Andi2ndDataSimulation
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter
from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter

#Andi 1
net = WavenetTCNSingleLevelAlphaPredicter(200,200,simulator=AndiDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()

net = WavenetTCNMultiTaskClassifierSingleLevelPredicter(200,200,simulator=AndiDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()

net = WavenetTCNSingleLevelChangePointPredicter(200,200,simulator=AndiDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()

#Andi 2
net = WavenetTCNSingleLevelAlphaPredicter(200,None,simulator=Andi2ndDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()

net = WavenetTCNMultiTaskClassifierSingleLevelPredicter(200,None,simulator=Andi2ndDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()

net = WavenetTCNSingleLevelChangePointPredicter(200,None,simulator=Andi2ndDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()

net = WavenetTCNSingleLevelDiffusionCoefficientPredicter(200,None,simulator=Andi2ndDataSimulation)
net.enable_early_stopping()
net.fit()
net.save_as_file()