import os
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import glob
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import confusion_matrix, f1_score
from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
import pandas as pd
from TheoreticalModels import RunAndTurnDiffusion
#from keras.metrics.confusion_metrics import AUC


class RunAndTurnSegmentator(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': True, 'epsilon': 1e-06, 'epochs':100}

    @classmethod
    def selected_hyperparameters(self):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': True, 'epsilon': 1e-06, 'epochs':100}

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [32, 64, 128, 256],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return [RunAndTurnDiffusion]

    @property
    def number_of_categories(self):
        return len(self.models_involved_in_predictive_model)

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories), verbose=0)

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_into_states(self, trajectories, to_hot_encoding=True)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_raw_trajectories(self, trajectories)

    def build_network(self):
        number_of_features = 2
        wavenet_filters = 32

        dilation_depth = 8
        initializer = 'he_normal'
        x1_kernel = 4
        x2_kernel = 2
        x3_kernel = 3
        x4_kernel = 10
        x5_kernel = 20

        dilation_depth = 8

        inputs = Input(shape=(None, number_of_features))

        x = WaveNetEncoder(wavenet_filters, dilation_depth, initializer=initializer)(inputs)

        x1 = convolutional_block(self, x, wavenet_filters, x1_kernel, [1,2,4], initializer)
        x2 = convolutional_block(self, x, wavenet_filters, x2_kernel, [1,2,4], initializer)
        x3 = convolutional_block(self, x, wavenet_filters, x3_kernel, [1,2,4], initializer)
        x4 = convolutional_block(self, x, wavenet_filters, x4_kernel, [1,4,8], initializer)

        x5 = Conv1D(filters=wavenet_filters, kernel_size=x5_kernel, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x5 = BatchNormalization()(x5)

        x = concatenate(inputs=[x1, x2, x3, x4, x5])

        x = Conv1D(filters=wavenet_filters*5, kernel_size=3, padding='causal', activation='relu', kernel_initializer=initializer)(x)
        output = Dense(units=2, activation='softmax', name='model_classification_output')(x)

        self.architecture = Model(inputs=inputs, outputs=output)

        optimizer = Adam(
            learning_rate=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )
        self.architecture.compile(optimizer=optimizer, loss=BinaryFocalCrossentropy(gamma=2, apply_class_balancing=True), metrics=['categorical_accuracy'])
        return self.architecture

    @property
    def type_name(self):
        return 'run_and_trap_segmentator'

    @property
    def dataset_type(self):
        return 'classifier'

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}"

    def plot_confusion_matrix(self, normalized=True):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predicted)

        if normalized:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        labels = ["Run", "Turn"]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        plt.title(f'Confusion Matrix (F1={round(f1_score(ground_truth, predicted, average="micro"),2)})')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        plt.show()