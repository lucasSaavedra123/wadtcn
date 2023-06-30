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
from .model_utils import transform_trajectories_into_displacements, convolutional_block, WaveNetEncoder, transform_trajectories_to_hurst_exponent

class WavenetTCNWithLSTMHurstExponentPredicter(PredictiveModel):
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
            'fbm_sub': {
                'lr': 0.0001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-6,
            },
            'fbm_brownian': {
                'lr': 0.0001,
                'batch_size': 16,
                'amsgrad': True,
                'epsilon': 1e-7,
            },
            'fbm_sup': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
            },
            'sbm_sub': {
                'lr': 0.0001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-6,
            },
            'sbm_brownian': {
                'lr': 0.0001,
                'batch_size': 16,
                'amsgrad': True,
                'epsilon': 1e-7,
            },
            'sbm_sup': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
            },
            'lw': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
            },
            'ctrw': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
            },
            'attm': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
            },
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[kwargs["model"]]
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
        return transform_trajectories_into_displacements(self, trajectories)

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length-1, 2))
        filters = 64
        dilation_depth = 8
        initializer = 'he_normal'

        x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

        x = convolutional_block(self, x, filters, 3, [1,2,4], initializer)

        x = Bidirectional(LSTM(units=filters, return_sequences=True, activation='tanh'))(x)
        x = Bidirectional(LSTM(units=filters//2, activation='tanh'))(x)

        x = Dense(units=128, activation='selu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
            lr=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    @property
    def type_name(self):
        return 'hurst_exponent'
