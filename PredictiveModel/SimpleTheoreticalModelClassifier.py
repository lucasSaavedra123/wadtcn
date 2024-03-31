import numpy as np
from tensorflow.keras.optimizers.legacy import Adam
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow import device, config

from .PredictiveModel import PredictiveModel
from CONSTANTS import *
from .model_utils import transform_trajectories_into_displacements, transform_trajectories_to_categorical_vector, TrackGenerator
from TheoreticalModels import HopDiffusion, FractionalBrownianMotion, TwoStateObstructedDiffusion

class SimpleTheoreticalModelClassifier(PredictiveModel):
    @classmethod
    def selected_hyperparameters(self):
        return {
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 100
        }

    @property
    def models_involved_in_predictive_model(self):
        return [FractionalBrownianMotion, TwoStateObstructedDiffusion, HopDiffusion]

    def default_hyperparameters(self):
        return {
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 100
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        pass

    def build_network(self):
        # Net filters and kernels
        initializer = 'he_normal'
        filters = 32

        inputs = Input(shape=(self.trajectory_length - 1, 2))
        x1 = Conv1D(filters=filters, kernel_size=3, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x1 = BatchNormalization()(x1)

        x2 = Conv1D(filters=filters, kernel_size=5, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x2 = BatchNormalization()(x2)

        x3 = Conv1D(filters=filters, kernel_size=7, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x3 = BatchNormalization()(x3)

        x_concat = concatenate(inputs=[x1, x2, x3])
        x = GlobalMaxPooling1D()(x_concat)

        dense_1 = Dense(units=512, activation='relu')(x)
        dense_2 = Dense(units=128, activation='relu')(dense_1)
        output_network = Dense(units=self.number_of_models_involved, activation='softmax')(dense_2)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
                        lr=self.hyperparameters['lr'],
                        amsgrad=self.hyperparameters['amsgrad'],
                        epsilon=self.hyperparameters['epsilon']
                    )

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
        return "original_theoretical_model_classifier"

    def fit(self):
        self.build_network()

        self.architecture.summary()

        if self.early_stopping:
            callbacks = [
                EarlyStopping(
                monitor="val_loss",
                min_delta=1e-3,
                patience=5,
                verbose=1,
                mode="min")
            ]
        else:
            callbacks = []

        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        X_train, Y_train = self.prepare_dataset(TRAINING_SET_SIZE_PER_EPOCH)
        X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH)

        with device(device_name):
            history_training_info = self.architecture.fit(
                X_train,
                Y_train,
                epochs=self.hyperparameters['epochs'],
                callbacks=callbacks,
                validation_data=[X_val, Y_val]
            ).history

        self.history_training_info = history_training_info
        self.trained = True
