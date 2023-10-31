import numpy as np
import pandas as pd

from TheoreticalModels.TwoStateObstructedDiffusion import TwoStateObstructedDiffusion


from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow import device, config
from keras.callbacks import EarlyStopping, Callback

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
            'epochs': 100,
            'batch_size': 16,
            'amsgrad': False,
            'epsilon': 1e-6,
        }

    @classmethod
    def selected_hyperparameters(self):
        return {
            'lr': 0.001,
            'epochs': 100,
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
        predicted = self.predict(trajectories).flatten()

        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predicted)

        if normalized:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        labels = ["No Obstructed", "Obstructed"]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        plt.title(f'Confusion Matrix')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        plt.show()

    def micro_f1_score(self, trajectories=None):
        if trajectories is None:
            trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)
        
        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()
        return f1_score(ground_truth, Y_predicted, average="micro")

    @property
    def type_name(self):
        return 'obstructed_trajectory_segmentator'

    def fit(self):
        if not self.trained:
            self.build_network()
            real_epochs = self.hyperparameters['epochs']
        else:
            real_epochs = self.hyperparameters['epochs'] - len(self.history_training_info['loss'])

        self.architecture.summary()

        if self.early_stopping:
            callbacks = [EarlyStopping(
                monitor="val_loss",
                min_delta=1e-3,
                patience=5,
                verbose=1,
                mode="min")]
        else:
            callbacks = []

        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        with device(device_name):
            X_train, Y_train = self.prepare_dataset(TRAINING_SET_SIZE_PER_EPOCH)
            X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH)

            history_training_info = self.architecture.fit(
                X_train, Y_train,
                epochs=real_epochs,
                callbacks=callbacks,
                validation_data=[X_val, Y_val],
                shuffle=True
            ).history

        if self.trained:
            for dict_key in history_training_info:
                self.history_training_info[dict_key] += history_training_info[dict_key]
        else:
            self.history_training_info = history_training_info
            self.trained = True
