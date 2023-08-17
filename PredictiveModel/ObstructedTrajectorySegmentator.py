import numpy as np
import pandas as pd

from TheoreticalModels.TwoStateObstructedDiffusion import TwoStateObstructedDiffusion


from tensorflow.keras.optimizers.legacy import Adam

from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

from .PredictiveModel import PredictiveModel
from .model_utils import build_segmentator_for, transform_trajectories_into_states, transform_trajectories_into_raw_trajectories
from CONSTANTS import *

class ObstructedTrajectorySegmentator(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return [TwoStateObstructedDiffusion]

    @property
    def number_of_categories(self):
        return len(self.models_involved_in_predictive_model)

    #These will be updated after hyperparameter search
    def default_hyperparameters(self):
        return {
            'lr': 0.001,
            'epochs': 5,
            'batch_size': 16,
            'amsgrad': False,
            'epsilon': 1e-6,
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 32, 128, 512, 1024],
            'epsilon': [1e-6, 1e-7, 1e-8],
        }

    def build_network(self):
        build_segmentator_for(self)

        optimizer = Adam(lr=self.hyperparameters['lr'],
                         epsilon=self.hyperparameters['epsilon'],
                         amsgrad=self.hyperparameters['amsgrad'])

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

        self.architecture.summary()

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        Y_predicted = self.architecture.predict(X)
        Y_predicted = (Y_predicted > 0.5).astype(int)
        return Y_predicted

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_into_states(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_raw_trajectories(self, trajectories)

    def plot_confusion_matrix(self, trajectories=None, normalized=True):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()

        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=Y_predicted)

        if normalized:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        labels = ["No Obstructed", "Obstructed"]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        # Plot matrix
        plt.title(f'Confusion Matrix')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        #plt.show()
        plt.savefig(str(self)+'.jpg')
        plt.clf()

    def __str__(self):
        return f"obstructed_trajectory_segmentator_length_{self.trajectory_length}"
