from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter
from CONSTANTS import *
from DataSimulation import Andi2ndDataSimulation, AndiDataSimulation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from Trajectory import Trajectory


simulators = [AndiDataSimulation, Andi2ndDataSimulation]

for simulator in simulators:
    network = WavenetTCNSingleLevelChangePointPredicter(200, 200, simulator=simulator)
    if simulator.STRING_LABEL == 'andi':
        network.load_as_file(selected_name='wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_999.h5')
        ts = simulator().simulate_segmentated_trajectories(12_500, 200, 200)
    else:
        network.load_as_file(selected_name='wavenet_changepoint_detector_200_None_andi2.h5')
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

    pred = network.predict(ts, apply_threshold=False).flatten()
    true = network.transform_trajectories_to_output(ts).flatten()
    fpr, tpr, thresholds = roc_curve(true, pred)
    roc_auc = auc(fpr, tpr)

    best_value = float('inf')
    for fpr_i, tpr_i, threshold_i in zip(fpr, tpr, thresholds):
        new_value = np.sqrt(((fpr_i)**2)+((1-tpr_i)**2))
        if new_value < best_value:
            threshold = threshold_i
            best_value = new_value
            pick_fpr, pick_tpr = fpr_i,tpr_i

    print(f'{simulator}->Selected threshold', threshold)
    pred = (pred > threshold).astype(int)
    cm = confusion_matrix(true, pred)
    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
