from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
import glob
import numpy as np
from Trajectory import Trajectory

from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter
from CONSTANTS import *
from DataSimulation import Andi2ndDataSimulation
from DataSimulation import AndiDataSimulation


FROM_ANDI_2 = True

if FROM_ANDI_2:
    alias_to_file = {
        'BCE 1/199': 'wavenet_changepoint_detector_200_None_andi2.h5',
    }
else:
    alias_to_file = {
        'BCE 1/5': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_5.h5',
        'BCE 1/10': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_10.h5',
        'BCE 1/20': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_20.h5',
        'BCE 1/50': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_50.h5',
        'BCE 1/999': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_999.h5',
        'Regularized BCE': 'wavenet_changepoint_detector_200_200.0_andi_with_bce_regularized.h5',
        'Penalized BCE': 'wavenet_changepoint_detector_200_200.0_andi_with_penalized_bce.h5',
    }

if FROM_ANDI_2:
    simulator = Andi2ndDataSimulation
    ts = []
    simulation_ts = Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_classification_training(12_500, 200, 200, True, 'val',enable_parallelism=True)

    for t in simulation_ts:
        sigma = np.random.uniform(0,2)

        ts.append(Trajectory(
            x=t.get_x().tolist(),
            y=t.get_y().tolist(),
            noise_x=np.random.randn(t.length) * sigma,
            noise_y=np.random.randn(t.length) * sigma,
            info=t.info,
        ))
else:
    simulator = AndiDataSimulation
    ts = simulator().simulate_segmentated_trajectories(12_500, 200, 200)
ax = None

for alias in alias_to_file:
    network = WavenetTCNSingleLevelChangePointPredicter(200, 200, simulator=simulator)
    network.load_as_file(selected_name=alias_to_file[alias])

    pred = network.predict(ts, apply_threshold=False).flatten()
    true = network.transform_trajectories_to_output(ts).flatten()

    fpr, tpr, thresholds = roc_curve(true, pred)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=alias)
        display.plot()
        ax = display.ax_
    else:
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=alias)
        display.plot(ax=ax)
plt.show()
