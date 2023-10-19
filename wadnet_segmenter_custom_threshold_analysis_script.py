import numpy as np
import ghostml
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from DataSimulation import CustomDataSimulation
from PredictiveModel.ImmobilizedTrajectorySegmentator import ImmobilizedTrajectorySegmentator


TRAIN_NEW_NETWORK = False

segmenter = ImmobilizedTrajectorySegmentator(25,0.25, simulator=CustomDataSimulation)

if TRAIN_NEW_NETWORK:
    segmenter.enable_early_stopping()
    segmenter.fit()
    segmenter.save_as_file()
else:
    segmenter.load_as_file()

print("Generating trayectories...")
trajectories = segmenter.simulator().simulate_trajectories_by_model(
    100_000,
    segmenter.trajectory_length,
    segmenter.trajectory_time,
    segmenter.models_involved_in_predictive_model
)

print("Predicting...")
ground_truth = segmenter.transform_trajectories_to_output(trajectories).flatten()
predictions = segmenter.predict(trajectories, apply_threshold=False).flatten()

print("GHOST analysis...")
thresholds = np.round(np.arange(0.05,0.95,0.025), 3)
threshold = ghostml.optimize_threshold_from_predictions(ground_truth, predictions, thresholds, ThOpt_metrics = 'ROC', N_subsets=100, subsets_size=0.2, with_replacement=False)
print("Selected Threshold:", threshold)