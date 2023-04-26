from DataSimulation import AndiDataSimulation, CustomDataSimulation
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier


WaveNetTCNFBMModelClassifier(25, 0.25, simulator=CustomDataSimulation).fit()