from DatabaseHandler import DatabaseHandler
from DataSimulation import CustomDataSimulation
from PredictiveModel.ImmobilizedTrajectorySegmentator import ImmobilizedTrajectorySegmentator
from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion')

ImmobilizedTrajectorySegmentator.analyze_hyperparameters(
    25,
    25 * EXPERIMENT_TIME_FRAME_BY_FRAME,
    initial_epochs=5,
    steps=5,
    simulator=CustomDataSimulation
)

DatabaseHandler.disconnect()
