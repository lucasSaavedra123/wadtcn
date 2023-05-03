from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.LSTMTheoreticalModelClassifier import LSTMTheoreticalModelClassifier
from PredictiveModel.LSTMAnomalousExponentPredicter import LSTMAnomalousExponentPredicter
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.OriginalTheoreticalModelClassifier import OriginalTheoreticalModelClassifier

INITIAL_LENGTH = 25
FINAL_LENGTH = 1000
STEP = 25


DatabaseHandler.connect_to_atlas('admin', 'admin', 'cluster0.9aachhp.mongodb.net')

for trajectory_length in range(INITIAL_LENGTH, FINAL_LENGTH + STEP, STEP):
    for predictive_model_class in [LSTMTheoreticalModelClassifier, LSTMAnomalousExponentPredicter, OriginalTheoreticalModelClassifier]:
        print("Model Class:", predictive_model_class, "Trajectory Length:", trajectory_length)
        model = predictive_model_class(trajectory_length, trajectory_length, simulator=AndiDataSimulation).fit()
        model.enable_data_persistance()
        model.enable_early_stopping()
        model.fit()
        model.save()

DatabaseHandler.disconnect()
