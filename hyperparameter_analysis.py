from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier

from TheoreticalModels import ALL_SUB_MODELS

TRAJECTORY_LENGTHS = [25, 50]

DatabaseHandler.connect_to_atlas('admin', 'admin', 'cluster0.9aachhp.mongodb.net')

for trajectory_length in TRAJECTORY_LENGTHS:
    for predictive_model_class in [
        WaveNetTCNTheoreticalModelClassifier,
        WaveNetTCNFBMModelClassifier,
        WaveNetTCNSBMModelClassifier,
    ]:
        predictive_model_class.analyze_hyperparameters(trajectory_length, trajectory_length, initial_epochs=5, steps=5, simulator=AndiDataSimulation)

    for class_model in ALL_SUB_MODELS:
        WavenetTCNWithLSTMHurstExponentPredicter(trajectory_length, trajectory_length, initial_epochs=5, steps=5, simulator=AndiDataSimulation, model=class_model.STRING_LABEL)

DatabaseHandler.disconnect()
