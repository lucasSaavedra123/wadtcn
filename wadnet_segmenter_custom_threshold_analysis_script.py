from sklearn.metrics import roc_auc_score
from DataSimulation import CustomDataSimulation
from PredictiveModel.ImmobilizedTrajectorySegmentator import ImmobilizedTrajectorySegmentator


TRAIN_NEW_NETWORK = True

segmenter = ImmobilizedTrajectorySegmentator(25,0.25, simulator=CustomDataSimulation)

if TRAIN_NEW_NETWORK:
    segmenter.enable_early_stopping()
    segmenter.fit()
    segmenter.save_as_file()
else:
    segmenter.load_as_file()

trajectories = segmenter.simulator().simulate_trajectories_by_model(
    12_500,
    segmenter.trajectory_length,
    segmenter.trajectory_time,
    segmenter.models_involved_in_predictive_model
)

ground_truth = segmenter.transform_trajectories_to_output(trajectories).flatten()
Y_predicted = segmenter.predict(trajectories, apply_threshold=False).flatten()

print(roc_auc_score(ground_truth, Y_predicted))