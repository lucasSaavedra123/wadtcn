import numpy as np
import ghostml
from DataSimulation import CustomDataSimulation
from PredictiveModel.ImmobilizedTrajectorySegmentator import ImmobilizedTrajectorySegmentator
from DatabaseHandler import DatabaseHandler

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

trained_networks = list(ImmobilizedTrajectorySegmentator.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=ImmobilizedTrajectorySegmentator.selected_hyperparameters()))
trained_networks = sorted(trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))
segmenter = trained_networks[0]
segmenter.enable_database_persistance()
segmenter.load_as_file()

DatabaseHandler.disconnect()

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
