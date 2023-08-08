import numpy as np
from tensorflow.keras.optimizers.legacy import Adam

from .PredictiveModel import PredictiveModel
from TheoreticalModels.ScaledBrownianMotion import ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionBrownian, ScaledBrownianMotionSuperDiffusive
from .model_utils import transform_trajectories_into_displacements, transform_trajectories_to_categorical_vector
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate
from keras.models import Model
from tensorflow.keras.utils import Sequence

import numpy as np
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate
from keras.callbacks import EarlyStopping
from tensorflow import device, config

from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_displacements, transform_trajectories_to_categorical_vector
from CONSTANTS import *

class TrackGenerator(Sequence):
    def __init__(self, batches, batch_size, dataset_function):
        self.batches = batches
        self.batch_size = batch_size
        self.dataset_function = dataset_function

    def __getitem__(self, item):
        tracks, classes = self.dataset_function(self.batch_size)
        return tracks, classes

    def __len__(self):
        return self.batches

class OriginalSBMModelClassifier(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return [ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionBrownian, ScaledBrownianMotionSuperDiffusive]

    def default_hyperparameters(self):
        return {
            'lr': 0.01,
            'batch_size': 16,
            'amsgrad': True,
            'epsilon': 1e-6,
            'epochs': 100
        }

    @classmethod
    def selected_hyperparameters(self):
        return {
            'lr': 0.01,
            'batch_size': 16,
            'amsgrad': True,
            'epsilon': 1e-6,
            'epochs': 100
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        pass

    def build_network(self):
        # Network filters and kernels
        initializer = 'he_normal'
        filters_size = 32
        x1_kernel_size = 4
        x2_kernel_size = 2
        x3_kernel_size = 3
        x4_kernel_size = 10
        x5_kernel_size = 6
        x6_kernel_size = 20

        inputs = Input(shape=(self.trajectory_length-1, 2))
        x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=2, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x1)
        x1 = BatchNormalization()(x1)
        x1 = GlobalMaxPooling1D()(x1)
        x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=2, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x2)
        x2 = BatchNormalization()(x2)
        x2 = GlobalMaxPooling1D()(x2)
        x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=2, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x3)
        x3 = BatchNormalization()(x3)
        x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x3)
        x3 = BatchNormalization()(x3)
        x3 = GlobalMaxPooling1D()(x3)
        x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x4)
        x4 = BatchNormalization()(x4)
        x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=8, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x4)
        x4 = BatchNormalization()(x4)
        x4 = GlobalMaxPooling1D()(x4)
        x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, padding='causal', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x5 = BatchNormalization()(x5)
        x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, dilation_rate=2, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x5)
        x5 = BatchNormalization()(x5)
        x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, dilation_rate=4, padding='causal',
                    activation='relu',
                    kernel_initializer=initializer)(x5)
        x5 = BatchNormalization()(x5)
        x5 = GlobalMaxPooling1D()(x5)
        x6 = Conv1D(filters=filters_size, kernel_size=x6_kernel_size, padding='same', activation='relu',
                    kernel_initializer=initializer)(inputs)
        x6 = BatchNormalization()(x6)
        x6 = GlobalMaxPooling1D()(x6)
        x_concat = concatenate(inputs=[x1, x2, x3, x4, x5, x6])
        dense_1 = Dense(units=615, activation='relu')(x_concat)
        dense_2 = Dense(units=150, activation='relu')(dense_1)
        output_network = Dense(units=self.number_of_models_involved, activation='softmax')(dense_2)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=self.hyperparameters['lr'],
                         epsilon=self.hyperparameters['epsilon'],
                         amsgrad=self.hyperparameters['amsgrad'])

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

    def type_name(self):
        return f"wavenet_tcn_sbm_model_classifier"

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

        with device(device_name):
            history_training_info = self.architecture.fit(
                X_train,
                Y_train,
                epochs=self.hyperparameters['epochs'],
                callbacks=callbacks,
                validation_data=TrackGenerator(VALIDATION_SET_SIZE_PER_EPOCH//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.prepare_dataset), shuffle=True
            ).history

        self.history_training_info = history_training_info
        self.trained = True
