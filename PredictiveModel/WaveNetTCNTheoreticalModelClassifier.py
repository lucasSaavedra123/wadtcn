import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate, Add
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow import make_ndarray

from .PredictiveModel import PredictiveModel
from TheoreticalModels import ANDI_MODELS, ALL_MODELS
from .model_utils import transform_trajectories_into_displacements, WaveNetEncoder, convolutional_block, transform_trajectories_to_categorical_vector
from tensorflow.keras import backend as K


class WaveNetTCNTheoreticalModelClassifier(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return ANDI_MODELS if self.simulator().STRING_LABEL == 'andi' else ALL_MODELS

    @property
    def number_of_models_involved(self):
        return len(self.models_involved_in_predictive_model)

    #These will be updated after hyperparameter search
    def default_hyperparameters(self):
        return {
            'training_set_size': 100000,
            'validation_set_size': 12500,
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 100,
            'with_early_stopping': True,
            'dropout_rate': 0
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 32, 128, 256, 512],
            'epsilon': [1e-6, 1e-7, 1e-8],
        }

    def conv_bloc(self, original_x, filters, kernel_size, dilation_rates, initializer):
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu', kernel_initializer=initializer, dilation_rate=dilation_rates[0])(original_x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rates[1], padding='causal', activation='relu', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rates[2], padding='causal', activation='relu', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)

        x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_initializer=initializer)(original_x)
        x_skip = BatchNormalization()(x_skip)

        x = Add()([x, x_skip])

        return x

    def build_network(self):
        # Net filters and kernels
        initializer = 'he_normal'
        filters = 64
        x1_kernel = 4
        x2_kernel = 2
        x3_kernel = 3
        x4_kernel = 10
        x5_kernel = 20

        dilation_depth = 8

        inputs = Input(shape=(self.trajectory_length-1, 2))

        x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

        x1 = convolutional_block(self, x, filters, x1_kernel, [1,2,4], initializer)
        x2 = convolutional_block(self, x, filters, x2_kernel, [1,2,4], initializer)
        x3 = convolutional_block(self, x, filters, x3_kernel, [1,2,4], initializer)
        x4 = convolutional_block(self, x, filters, x4_kernel, [1,4,8], initializer)

        x5 = Conv1D(filters=filters, kernel_size=x5_kernel, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x5 = BatchNormalization()(x5)

        x = concatenate(inputs=[x1, x2, x3, x4, x5])

        x = GlobalMaxPooling1D()(x)

        dense_1 = Dense(units=512, activation='relu')(x)
        dense_2 = Dense(units=128, activation='relu')(dense_1)
        output_network = Dense(units=self.number_of_models_involved, activation='softmax')(dense_2)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=self.hyperparameters['lr'],
                         amsgrad=self.hyperparameters['amsgrad'],
                         epsilon=self.hyperparameters['epsilon'])

        self.architecture.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        Y_predicted = self.architecture.predict(X)
        Y_predicted = np.argmax(Y_predicted, axis=-1)
        return Y_predicted

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_categorical_vector(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_displacements(self, trajectories)

    @property
    def type_name(self):
        return "wavenet_tcn_theoretical_model_classifier"