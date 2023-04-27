from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.LSTMTheoreticalModelClassifier import LSTMTheoreticalModelClassifier

DatabaseHandler.connect_to_atlas('admin', 'admin', 'cluster0.9aachhp.mongodb.net')

TRAJECTORY_LENGTHS = [25, 50]

for predictive_model_class in [
    WaveNetTCNFBMModelClassifier,
    WaveNetTCNSBMModelClassifier,
    WavenetTCNWithLSTMHurstExponentPredicter,
    WaveNetTCNTheoreticalModelClassifier,
    LSTMTheoreticalModelClassifier
]:
    for trajectory_length in TRAJECTORY_LENGTHS:
        predictive_model_class.analyze_hyperparameters(trajectory_length, trajectory_length, initial_epochs=5, steps=5, simulator=AndiDataSimulation)

DatabaseHandler.disconnect()
