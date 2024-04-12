import numpy as np
from keras.layers import Dense, Input, LSTM, Bidirectional, Flatten, TimeDistributed
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


class WavenetTCNWithLSTMModelSingleLevelPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        hyperparameters = {'lr': 0.0001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 100}
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
        return transform_trajectories_to_single_level_model(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)
        return X

    def build_network(self):
        if self.wadnet_tcn_encoder is None:
            number_of_features = 2
            inputs = Input(shape=(self.trajectory_length, number_of_features))
            wavenet_filters = 16
            dilation_depth = 8
            initializer = 'he_normal'

            x = WaveNetEncoder(wavenet_filters, dilation_depth, initializer=initializer)(inputs)
            unet_1 = Unet((self.trajectory_length, wavenet_filters), '1d', 2, unet_index=1)(x)
            unet_2 = Unet((self.trajectory_length, wavenet_filters), '1d', 3, unet_index=2)(x)
            unet_3 = Unet((self.trajectory_length, wavenet_filters), '1d', 4, unet_index=3)(x)
            unet_4 = Unet((self.trajectory_length, wavenet_filters), '1d', 9, unet_index=4)(x)

            #output_network = Average()([unet_1, unet_2, unet_3, unet_4])
            
            x = concatenate([unet_1, unet_2, unet_3, unet_4])
            x = Conv1D(1, 3, 1, padding='same', activation='sigmoid')(x)
            output_network = TimeDistributed(Dense(units=4, activation='softmax'))(x)

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

        self.architecture.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    @property
    def type_name(self):
        return 'wavenet_single_level_model'

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

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}"