from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotion
from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *

class WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter(PredictiveModel):
    def default_hyperparameters(self):
        return {
            'lr': 0.0001,
            'batch_size': 8,
            'amsgrad': False,
            'epsilon': 1e-08,
            'epochs': 100
        }

    @classmethod
    def selected_hyperparameters(self):
        return {
            'lr': 0.0001,
            'batch_size': 8,
            'amsgrad': False,
            'epsilon': 1e-08,
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
        return transform_trajectories_into_squared_differences(self, trajectories, normalize=True)

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length-1, 1))
        filters = 64
        dilation_depth = 8
        initializer = 'he_normal'

        x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)
        x = convolutional_block(self, x, filters, 3, [1,2,4], initializer)
        x = GlobalAveragePooling1D()(x)

        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=128, activation='relu')(x)
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

        plot_predicted_and_ground_truth_histogram(ground_truth, predicted, range=[[0,1],[0,1]])
