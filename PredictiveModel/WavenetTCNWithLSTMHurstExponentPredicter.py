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
from .model_utils import *
from CONSTANTS import *


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
            'fbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'fbm_brownian': {'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 100},
            'fbm_sup': {'lr': 0.0001, 'batch_size': 8, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'sbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'sbm_brownian': {'lr': 0.001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-07, 'epochs': 100},
            'sbm_sup': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 100},
            'lw': {'lr': 0.001, 'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 100},
            'ctrw': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 100},
            'attm': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 100},
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[kwargs["model"]]
        hyperparameters['epochs'] = 100

        return hyperparameters

    @classmethod
    def selected_hyperparameters(self, model_label):
        model_string_to_hyperparameters_dictionary = {
            'fbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'fbm_brownian': {'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 100},
            'fbm_sup': {'lr': 0.0001, 'batch_size': 8, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'sbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'sbm_brownian': {'lr': 0.001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-07, 'epochs': 100},
            'sbm_sup': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 100},
            'lw': {'lr': 0.001, 'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 100},
            'ctrw': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 100},
            'attm': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 100},
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
        X = transform_trajectories_into_displacements(self, trajectories) if self.simulator.STRING_LABEL == 'andi' else transform_trajectories_into_displacements_with_time(self, trajectories)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)
        return X

    def build_network(self):
        if self.wadnet_tcn_encoder is None:
            number_of_features = 2 if self.simulator.STRING_LABEL == 'andi' else 3
            inputs = Input(shape=(self.trajectory_length-1, number_of_features))
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

        else:
            inputs = Input(shape=(64))
            x = Dense(units=128, activation='selu')(inputs)
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
        return 'wavenet_hurst_exponent'

    def plot_bias(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten() * 2
        predicted = self.predict(trajectories).flatten() * 2

        plot_bias(ground_truth, predicted, symbol='alpha')

    def plot_predicted_and_ground_truth_distribution(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten() * 2
        predicted = self.predict(trajectories).flatten() * 2

        plot_predicted_and_ground_truth_distribution(ground_truth, predicted)

    def plot_predicted_and_ground_truth_histogram(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten() * 2
        predicted = self.predict(trajectories).flatten() * 2

        plot_predicted_and_ground_truth_histogram(ground_truth, predicted, range=[[0,2],[0,2]])
