import numpy as np
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, Bidirectional
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

from TheoreticalModels.AnnealedTransientTimeMotion import AnnealedTransientTimeMotion
from TheoreticalModels.ContinuousTimeRandomWalk import ContinuousTimeRandomWalk
from TheoreticalModels.LevyWalk import LevyWalk
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionBrownian, FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionSuperDiffusive
from TheoreticalModels.ScaledBrownianMotion import ScaledBrownianMotionBrownian, ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionSuperDiffusive
from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_displacements, convolutional_block, WaveNetEncoder, transform_trajectories_into_raw_trajectories
from TheoreticalModels import ANDI_MODELS

class WavenetTCNWithLSTMHurstExponentPredicter(PredictiveModel):
    #These will be updated after hyperparameter search
    def default_hyperparameters(self):
        return {
            'training_set_size': 100000,
            'validation_set_size': 12500,
            'with_early_stopping': False,
            'fbm_sub': {
                'lr': 0.0001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-6,
                'model': FractionalBrownianMotionSubDiffusive
            },
            'fbm_brownian': {
                'lr': 0.0001,
                'batch_size': 16,
                'amsgrad': True,
                'epsilon': 1e-7,
                'model': FractionalBrownianMotionBrownian
            },
            'fbm_sup': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': FractionalBrownianMotionSuperDiffusive
            },
            'sbm_sub': {
                'lr': 0.0001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-6,
                'model': ScaledBrownianMotionSubDiffusive
            },
            'sbm_brownian': {
                'lr': 0.0001,
                'batch_size': 16,
                'amsgrad': True,
                'epsilon': 1e-7,
                'model': ScaledBrownianMotionBrownian
            },
            'sbm_sup': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': ScaledBrownianMotionSuperDiffusive
            },
            'lw': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': LevyWalk
            },
            'ctrw': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': ContinuousTimeRandomWalk
            },
            'attm': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': AnnealedTransientTimeMotion
            },
            'epochs': 5
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [16, 32, 64, 128],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return [self.hyperparameters[self.extra_parameters["model"]]['model']]

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        return self.architecture.predict(X)

    def transform_trajectories_to_output(self, trajectories):
        Y = np.empty((len(trajectories), 1))

        for index, trajectory in enumerate(trajectories):
            Y[index, 0] = trajectory.hurst_exponent()

        return Y

    def transform_trajectories_to_input(self, trajectories):
        """
        Para FBM es mejor la trayectoria en sí
        Para ATTM es mejor los desplazamientos
        """
        return transform_trajectories_into_displacements(self, trajectories)

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length-1, 2))
        filters = 64
        dilation_depth = 8
        initializer = 'he_normal'

        x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

        x = convolutional_block(self, x, filters, 5, [1,2,4], initializer)
        #x = GlobalMaxPooling1D()(x)

        x = Bidirectional(LSTM(units=filters, return_sequences=True, activation='tanh'))(x)
        x = Bidirectional(LSTM(units=filters//2, activation='tanh'))(x)

        x = Dense(units=128, activation='selu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
            lr=self.hyperparameters[self.extra_parameters["model"]]['lr'],
            epsilon=self.hyperparameters[self.extra_parameters["model"]]['epsilon'],
            amsgrad=self.hyperparameters[self.extra_parameters["model"]]['amsgrad']
        )

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    @property
    def type_name(self):
        return 'hurst_exponent'
