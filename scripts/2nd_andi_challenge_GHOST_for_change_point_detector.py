from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter
import ghostml
from CONSTANTS import *
from DataSimulation import Andi2ndDataSimulation
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

network = WavenetTCNSingleLevelChangePointPredicter(200, None, simulator=Andi2ndDataSimulation)
network.load_as_file()

ts = []
for i in range(5):
   ts += Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_classification_training(TRAINING_SET_SIZE_PER_EPOCH,200,None,True,f'train_{i}', ignore_boundary_effects=True, enable_parallelism=True, type_of_simulation='models_phenom')

pred = network.predict(ts, apply_threshold=False).flatten()
true = network.transform_trajectories_to_output(ts).flatten()

count = Counter(true)
positive_is_majority = count[1] > count[0]

if positive_is_majority:
    true = 1 - np.array(true)
    pred = 1 - np.array(pred)

thresholds = np.round(np.arange(0.05,0.95,0.025), 3)
threshold = ghostml.optimize_threshold_from_predictions(true, pred, thresholds, ThOpt_metrics = 'ROC', N_subsets=100, subsets_size=0.0001, with_replacement=False)
print('Selected threshold', threshold)
pred = (pred > threshold).astype(int)
cm = confusion_matrix(true, pred)
cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
