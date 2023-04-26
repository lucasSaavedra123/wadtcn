import numpy as np
from keras.layers import Dense, Input, LSTM, Conv1D, Bidirectional, BatchNormalization, Add, Multiply
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from TheoreticalModels.AnnealedTransientTimeMotion import AnnealedTransientTimeMotion
from TheoreticalModels.ContinuousTimeRandomWalk import ContinuousTimeRandomWalk
from TheoreticalModels.LevyWalk import LevyWalk
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionBrownian, FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionSuperDiffusive
from TheoreticalModels.ScaledBrownianMotion import ScaledBrownianMotionBrownian, ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionSuperDiffusive
from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_displacements, convolutional_block

class HurstExponentModel(PredictiveModel):

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
        return transform_trajectories_into_displacements(self, trajectories)

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length-1, 2))
        dilation_depth = 8
        initializer = 'he_normal'
        filters = 64

        wavenet_dilations = [2**i for i in range(dilation_depth)]
        conv_1d_tanh = [Conv1D(filters, kernel_size=3, dilation_rate=dilation, padding='causal', activation='tanh') for dilation in wavenet_dilations]
        conv_1d_sigm = [Conv1D(filters, kernel_size=3, dilation_rate=dilation, padding='causal', activation='sigmoid') for dilation in wavenet_dilations]

        x = Conv1D(filters, 3, padding='causal')(inputs)

        layers_to_add = [x]

        for i in range(dilation_depth):
            tanh_out = conv_1d_tanh[i](x)
            sigm_out = conv_1d_sigm[i](x)

            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters, 1, padding='causal')(x)

            layers_to_add.append(x)

        x = Add()(layers_to_add)
        x = BatchNormalization()(x)

        x = convolutional_block(self, x, filters, 5, [1,2,4], initializer)

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