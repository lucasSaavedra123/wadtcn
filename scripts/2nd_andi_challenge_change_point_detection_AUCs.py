from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter
import ghostml
import os
from CONSTANTS import *
from DataSimulation import Andi2ndDataSimulation
from DataSimulation import AndiDataSimulation
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc

alias_to_file = {
    'BCE 1/5': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_5.h5',
    'BCE 1/10': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_10.h5',
    'BCE 1/20': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_20.h5',
    'BCE 1/50': 'wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_50.h5',
    'Regularized BCE': 'wavenet_changepoint_detector_200_200.0_andi_with_bce_regularized.h5',
    'Penalized BCE': 'wavenet_changepoint_detector_200_200.0_andi_with_penalized_bce.h5',
}

print("Generate trajectories...")
ts = AndiDataSimulation().simulate_segmentated_trajectories(1_000, 200, 200)
ax = None
#ts = []
#for i in range(1):
#   ts += AndiDataSimulation().simulate_phenomenological_trajectories_for_classification_training(TRAINING_SET_SIZE_PER_EPOCH,200,None,True,f'train_{i}', ignore_boundary_effects=True, enable_parallelism=True, type_of_simulation='models_phenom')

for alias in alias_to_file:
    network = WavenetTCNSingleLevelChangePointPredicter(200, 200, simulator=AndiDataSimulation)
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
