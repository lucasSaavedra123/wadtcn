from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotion
from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *

class WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter(PredictiveModel):
    def default_hyperparameters(self):
        return {
            'lr': 0.0001,
            'batch_size': 64,
            'amsgrad': True,
            'epsilon': 1e-07,
            'epochs': 100
        }

    @classmethod
    def selected_hyperparameters(self):
        return {
            'lr': 0.0001,
            'batch_size': 64,
            'amsgrad': True,
            'epsilon': 1e-07,
            'epochs': 100
        }

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
        return [FractionalBrownianMotion]

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories))

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_diffusion_coefficient(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_displacements_with_time(self, trajectories, normalize=False)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)

        return X

    def build_network(self):
        if self.wadnet_tcn_encoder is None:
            inputs = Input(shape=(self.trajectory_length-1, 3))
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
            learning_rate=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    @property
    def type_name(self):
        return 'wavenet_diffusion_coefficient'

    def plot_bias(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_bias(ground_truth, predicted, symbol='d')

    def plot_predicted_and_ground_truth_distribution(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_predicted_and_ground_truth_distribution(ground_truth, predicted)

    def plot_predicted_and_ground_truth_histogram(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_predicted_and_ground_truth_histogram(ground_truth, predicted, a_range=[[0,1], [0,1]])
