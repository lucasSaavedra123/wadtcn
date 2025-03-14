import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, auc

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
        'BCE 1/199': 'wavenet_changepoint_detector_200_200.0_andi.h5',
        'Regularized BCE': 'wavenet_changepoint_detector_200_200.0_andi_with_bce_regularized.h5',
        'Penalized BCE': 'wavenet_changepoint_detector_200_200.0_andi_with_penalized_bce.h5',
    }

print("Generate trajectories...")

if FROM_ANDI_2:
    simulator = Andi2ndDataSimulation
    ts = simulator().simulate_phenomenological_trajectories_for_classification_training(12_500, 200, 200, get_from_cache=False, enable_parallelism=True)
else:
    simulator = AndiDataSimulation
    ts = simulator().simulate_segmentated_trajectories(12_000, 200, 200)
ax = None

for alias in alias_to_file:
    network = WavenetTCNSingleLevelChangePointPredicter(200, 200 if not FROM_ANDI_2 else None, simulator=simulator)
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
