from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier

from TheoreticalModels import ALL_SUB_MODELS

DatabaseHandler.connect_to_local('anomalous_diffusion')

for predictive_model_class in [
    WaveNetTCNTheoreticalModelClassifier,
    WaveNetTCNFBMModelClassifier,
    WaveNetTCNSBMModelClassifier
]:
    predictive_model_class.analyze_hyperparameters(25, 25, initial_epochs=5, steps=5, simulator=AndiDataSimulation)

for class_model in ALL_SUB_MODELS:
    WavenetTCNWithLSTMHurstExponentPredicter(25, 25, initial_epochs=5, steps=5, simulator=AndiDataSimulation, model=class_model.STRING_LABEL)

DatabaseHandler.disconnect()
