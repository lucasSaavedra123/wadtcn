import numpy as np
from keras.layers import Dense, Input, LSTM, Bidirectional
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import tqdm


from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotion
from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_raw_trajectories, transform_trajectories_to_hurst_exponent, transform_trajectories_to_mean_square_displacement_segments

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

from .model_utils import *
from CONSTANTS import *

class SlidingWindowDiffusionCoefficientPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def model_string_to_class_dictionary(self):
        return {
            'fbm': { 'model': FractionalBrownianMotion },
        }

    def default_hyperparameters(self, **kwargs):
        model_string_to_hyperparameters_dictionary = {
            'fbm': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[kwargs["model"]]
        hyperparameters['epochs'] = 100

        return hyperparameters

    @classmethod
    def selected_hyperparameters(self, model_label):
        model_string_to_hyperparameters_dictionary = {
            'fbm': {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
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

    def prepare_dataset(self, set_size):
        trajectories = self.simulator().simulate_trajectories_by_model(set_size, self.trajectory_length*self.extension_factor, self.trajectory_time*self.extension_factor, self.models_involved_in_predictive_model)

        trajectories = self.adjust_trajectories_for_training_and_validation(trajectories)

        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

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

        #x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

        #x = convolutional_block(self, x, filters, 3, [1,2,4], initializer)

        x = convolutional_block(self, inputs, filters, 3, [1,2,4], initializer)

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
    def extension_factor(self):
        return 10

    @property
    def type_name(self):
        return f'segment_diffusion_coefficient_{self.extra_parameters["model"]}'

    def plot_bias(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length*self.extension_factor, self.trajectory_time*self.extension_factor, self.models_involved_in_predictive_model)
        trajectories = self.adjust_trajectories_for_training_and_validation(trajectories)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_bias(ground_truth, predicted, symbol='d')

    def plot_predicted_and_ground_truth_distribution(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length*self.extension_factor, self.trajectory_time*self.extension_factor, self.models_involved_in_predictive_model)
        trajectories = self.adjust_trajectories_for_training_and_validation(trajectories)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_predicted_and_ground_truth_distribution(ground_truth, predicted)

    def plot_predicted_and_ground_truth_histogram(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length*self.extension_factor, self.trajectory_time*self.extension_factor, self.models_involved_in_predictive_model)
        trajectories = self.adjust_trajectories_for_training_and_validation(trajectories)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_predicted_and_ground_truth_histogram(ground_truth, predicted)

    def adjust_trajectories_for_training_and_validation(self, trajectories):
        for trajectory_index in range(len(trajectories)):
            initial_index = np.random.randint(trajectories[trajectory_index].length-self.trajectory_length)
            trajectories[trajectory_index] = trajectories[trajectory_index].build_noisy_subtrajectory_from_range(initial_index, initial_index+self.trajectory_length)
            assert trajectories[trajectory_index].length == self.trajectory_length

        return trajectories

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
        X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH)

        with device(device_name):
            history_training_info = self.architecture.fit(
                X_train,
                Y_train,
                epochs=self.hyperparameters['epochs'],
                callbacks=callbacks,
                validation_data=[X_val, Y_val], shuffle=True
            ).history

        self.history_training_info = history_training_info
        self.trained = True
