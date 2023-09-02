import numpy as np
from keras.layers import Dense, Input, LSTM, Bidirectional
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

from TheoreticalModels.AnnealedTransientTimeMotion import AnnealedTransientTimeMotion
from TheoreticalModels.ContinuousTimeRandomWalk import ContinuousTimeRandomWalk
from TheoreticalModels.LevyWalk import LevyWalk
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionBrownian, FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionSuperDiffusive
from TheoreticalModels.ScaledBrownianMotion import ScaledBrownianMotionBrownian, ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionSuperDiffusive
from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_raw_trajectories, transform_trajectories_to_hurst_exponent

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

from model_utils import *
from CONSTANTS import *

class OriginalHurstExponentPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def model_string_to_class_dictionary(self):
        return {
            'fbm_sub': { 'model': FractionalBrownianMotionSubDiffusive },
            'fbm_brownian': { 'model': FractionalBrownianMotionBrownian },
            'fbm_sup': { 'model': FractionalBrownianMotionSuperDiffusive },
            'sbm_sub': { 'model': ScaledBrownianMotionSubDiffusive },
            'sbm_brownian': {'model': ScaledBrownianMotionBrownian },
            'sbm_sup': { 'model': ScaledBrownianMotionSuperDiffusive},
            'lw': {'model': LevyWalk },
            'ctrw': { 'model': ContinuousTimeRandomWalk },
            'attm': {'model': AnnealedTransientTimeMotion}
        }

    def default_hyperparameters(self, **kwargs):
        model_string_to_hyperparameters_dictionary = {
            'fbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
            'fbm_brownian': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
            'fbm_sup': { 'lr': 0.001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-7, 'epochs': 100},
            'sbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
            'sbm_brownian': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
            'sbm_sup': { 'lr': 0.001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-7, 'epochs': 100},
            'lw': { 'lr': 0.001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-7, 'epochs': 100},
            'ctrw': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
            'attm': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[kwargs["model"]]
        hyperparameters['epochs'] = 100

        return hyperparameters

    @classmethod
    def selected_hyperparameters(self, model_label):
        model_string_to_hyperparameters_dictionary = {
            'fbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
            'fbm_brownian': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
            'fbm_sup': { 'lr': 0.001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-7, 'epochs': 100},
            'sbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
            'sbm_brownian': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
            'sbm_sup': { 'lr': 0.001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-7, 'epochs': 100},
            'lw': { 'lr': 0.001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-7, 'epochs': 100},
            'ctrw': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
            'attm': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-6, 'epochs': 100},
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[model_label]
        hyperparameters['epochs'] = 100

        return hyperparameters

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 16, 32, 64],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return [self.model_string_to_class_dictionary()[self.extra_parameters["model"]]['model']]

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories))

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_hurst_exponent(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_raw_trajectories(self, trajectories, normalize=True)

    def build_network(self):
        inputs = Input(shape=(self.track_length, 2))
        x = LSTM(units=64, return_sequences=True, input_shape=(2, self.track_length))(inputs)
        x = LSTM(units=16)(x)
        x = Dense(units=128, activation='selu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)

        self.arquitecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(lr=self.net_params[self.fbm_type]['lr'], epsilon=self.net_params[self.fbm_type]['epsilon'],
                         amsgrad=self.net_params[self.fbm_type]['amsgrad'])


        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    @property
    def type_name(self):
        return 'original_hurst_exponent'

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
