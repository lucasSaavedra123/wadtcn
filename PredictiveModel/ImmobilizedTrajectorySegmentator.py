import numpy as np
import pandas as pd

from TheoreticalModels.TwoStateImmobilizedDiffusion import TwoStateImmobilizedDiffusion

from tensorflow.keras.optimizers.legacy import Adam

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from CONSTANTS import *

from .PredictiveModel import PredictiveModel
from .model_utils import build_segmentator_for, transform_trajectories_into_raw_trajectories, transform_trajectories_into_displacements_with_time, transform_trajectories_into_states, build_wavenet_tcn_segmenter_from_encoder_for

class ImmobilizedTrajectorySegmentator(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return [TwoStateImmobilizedDiffusion]

    @property
    def number_of_categories(self):
        return None

    def default_hyperparameters(self):
        return {
            'lr': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-6,
        }

    @classmethod
    def selected_hyperparameters(self):
        return {
            'lr': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-6,
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 16, 32, 64],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    def build_network(self):
        if self.wadnet_tcn_encoder is None:
            build_segmentator_for(self, with_wadnet=True, number_of_features=2, filters=32, input_size=self.trajectory_length, with_skip_connections=True)
        else:
            build_wavenet_tcn_segmenter_from_encoder_for(self, 64)

        optimizer = Adam(lr=self.hyperparameters['lr'],
                         epsilon=self.hyperparameters['epsilon'],
                         amsgrad=self.hyperparameters['amsgrad'])

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae', 'accuracy'])

    def predict(self, trajectories, apply_threshold=True):
        X = self.transform_trajectories_to_input(trajectories)
        Y_predicted = self.architecture.predict(X)

        if apply_threshold:
            Y_predicted = (Y_predicted > self.selected_threshold).astype(int)

        return Y_predicted

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_into_states(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories, normalize=True)
        #X = transform_trajectories_into_displacements_with_time(self, trajectories)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)

        return X

    def plot_confusion_matrix(self, normalized=True):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predicted)

        if normalized:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        labels = ["Free Diffusion", "Immovilized Diffusion"]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        plt.title(f'Confusion Matrix (F1={round(f1_score(ground_truth, predicted, average="micro"),2)})')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        plt.show()

    @property
    def selected_threshold(self):
        return 0.5

    @property
    def type_name(self):
        return "immobilized_trajectory_segmentator"

    def micro_f1_score(self, trajectories=None):
        if trajectories is None:
            trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)
        
        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()
        return f1_score(ground_truth, Y_predicted, average="micro")