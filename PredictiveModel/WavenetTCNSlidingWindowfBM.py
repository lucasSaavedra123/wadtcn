from random import shuffle

from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import device, config
from keras.callbacks import EarlyStopping, Callback
from keras.layers import MultiHeadAttention

from Trajectory import Trajectory
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotion
from TheoreticalModels.BrownianMotion import BrownianMotion
from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *

class WavenetTCNSlidingWindowfBM(PredictiveModel):
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
        return [BrownianMotion]

    def predict(self, trajectories):
        results = []

        for trajectory in trajectories:
            matrix = np.empty((trajectory.length-self.trajectory_length+1, trajectory.length))
            matrix[:] = np.nan
            for index in range(0,trajectory.length-self.trajectory_length+1,1):
                sub_trajectory = trajectory.build_noisy_subtrajectory_from_range(index, index+self.trajectory_length)
                matrix[index,index:index+self.trajectory_length] = 10**self.architecture.predict(self.transform_trajectories_to_input([sub_trajectory]), verbose=0)

            results.append(np.nanmean(matrix, axis=0).tolist())

        return results

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_diffusion_coefficient(self, trajectories, transformation=lambda x: np.log10(x))

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_displacements(self, trajectories, normalize=False)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)

        return X

    def simulate_trajectories(self, set_size, sample_from_ds=False, same_length=True):
        trajectories = []

        while len(trajectories) != set_size:
            new_d = np.random.uniform(10**-3,10**3) if sample_from_ds else 10**np.random.choice(np.logspace(-3,3,1000))

            new_length = self.trajectory_length * np.random.randint(1,10)
            
            simulation_result = BrownianMotion(new_d).custom_simulate_rawly(new_length, None)

            new_trajectory = Trajectory(
                simulation_result['x'],
                simulation_result['y'],
                t=simulation_result['t'],
                noise_x=simulation_result['x_noisy']-simulation_result['x'],
                noise_y=simulation_result['y_noisy']-simulation_result['y'],
                exponent_type=simulation_result['exponent_type'],
                exponent=simulation_result['exponent'],
                model_category=self,
                info=simulation_result['info']
            )

            if same_length:
                initial_index = np.random.randint(0, new_length-self.trajectory_length+1)
                new_trajectory = new_trajectory.build_noisy_subtrajectory_from_range(initial_index, initial_index+self.trajectory_length)          
            
            trajectories.append(new_trajectory)

        shuffle(trajectories)

        return trajectories

    def prepare_dataset(self, set_size):
        trajectories = self.simulate_trajectories(set_size)
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def build_network(self):
        if self.wadnet_tcn_encoder is None:
            inputs = Input(shape=(self.trajectory_length-1, 2))
            filters = 64
            dilation_depth = 8
            initializer = 'he_normal'

            x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

            x = convolutional_block(self, x, filters, 3, [1,2,4], initializer)

            x = Bidirectional(LSTM(units=filters, return_sequences=True, activation='tanh'))(x)
            x = Bidirectional(LSTM(units=filters//2, activation='tanh'))(x)

            x = Dense(units=128, activation='selu')(x)

            def custom_activation(x):
                return K.tanh(x) * 3.1

            output_network = Dense(units=1, activation=Activation(custom_activation))(x)

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
        return 'wavenet_diffusion_coefficient'

    def plot_bias(self):
        trajectories = self.simulate_trajectories(VALIDATION_SET_SIZE_PER_EPOCH)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.architecture.predict(self.transform_trajectories_to_input(trajectories)).flatten()

        plot_bias(ground_truth, predicted, symbol='d')

    def plot_predicted_and_ground_truth_distribution(self):
        trajectories = self.simulate_trajectories(VALIDATION_SET_SIZE_PER_EPOCH, sample_from_ds=False)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.architecture.predict(self.transform_trajectories_to_input(trajectories)).flatten()

        plot_predicted_and_ground_truth_distribution(ground_truth, predicted)

    def plot_predicted_and_ground_truth_histogram(self):
        trajectories = self.simulate_trajectories(VALIDATION_SET_SIZE_PER_EPOCH, sample_from_ds=False)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.architecture.predict(self.transform_trajectories_to_input(trajectories)).flatten()

        plot_predicted_and_ground_truth_histogram(ground_truth, predicted)
