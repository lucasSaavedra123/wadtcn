import numpy as np
from keras.layers import Dense, Input, LSTM, Bidirectional
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import tqdm
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
from Trajectory import Trajectory


from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotion
from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_raw_trajectories, transform_trajectories_to_hurst_exponent, transform_trajectories_to_mean_square_displacement_segments

from .PredictiveModel import PredictiveModel
from TheoreticalModels.ScaledBrownianMotion import ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionBrownian, ScaledBrownianMotionSuperDiffusive
from .model_utils import transform_trajectories_into_displacements, transform_trajectories_to_categorical_vector
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate, Activation
from keras.models import Model
from tensorflow.keras.utils import Sequence
from keras import backend as K
from andi_datasets.datasets_challenge import datasets_phenom
import numpy as np
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate
from keras.callbacks import EarlyStopping
from tensorflow import device, config

from .model_utils import *
from CONSTANTS import *

def detect_change_points(list_of_values):
    cps = []
    for i in range(1, len(list_of_values)):
        if list_of_values[i-1] != list_of_values[i]:
            cps.append(i)
    return cps

class SlidingWindowDiffusionCoefficientPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def model_string_to_class_dictionary(self):
        return {
            'fbm': { 'model': FractionalBrownianMotion },
        }

    def default_hyperparameters(self, **kwargs):
        model_string_to_hyperparameters_dictionary = {
            'fbm': {'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[kwargs["model"]]
        hyperparameters['epochs'] = 100

        return hyperparameters

    @classmethod
    def selected_hyperparameters(self, model_label):
        model_string_to_hyperparameters_dictionary = {
            'fbm': {'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-7, 'epochs': 100},
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

    def simulate_trajectories(self, set_size):
        trajectories = []

        ds = np.random.choice(np.logspace(-3,3,10000), size=set_size, replace=True)
        alphas = np.random.uniform(0,2,size=set_size)
        
        for i in list(range(set_size)):
            d = ds[i]
            a = alphas[i]
            trajs, labels = datasets_phenom().create_dataset(
                N_model = 1,
                T = self.trajectory_length,
                dics=[{'model': 'single_state', 'Ds': [d, 1e-99], 'alphas' : [a, 1e-99]}]
            )

            trajectories+=Trajectory.transform_dataset_from_create_dataset_to_trajectories(trajs, labels)

        return trajectories

    def prepare_dataset(self, set_size):
        """
        num_fovs = 1

        experiments = np.random.randint(1,5+1,size=100)

        dics = []

        for e in experiments:
            new_dic = _get_dic_andi2(e)
            new_dic['T'] = self.trajectory_length
            new_dic['N'] = 1000
            dics.append(new_dic)

        df_fov, _ , lab_e = challenge_phenom_dataset(
            experiments = experiments,
            dics = dics,
            num_fovs =num_fovs,
            return_timestep_labs = True
        )

        trajectories = [t for t in Trajectory.transform_dataset_from_challenge_phenom_dataset_to_trajectories(df_fov)]
        #trajectories = self.adjust_trajectories_for_training_and_validation(trajectories)
        print("Number of trajectories:", len(trajectories))
        """
        trajectories = self.simulate_trajectories(set_size)
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories))

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_diffusion_coefficient(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_displacements(self, trajectories)

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length-1, 2))
        filters = 64
        dilation_depth = 8
        initializer = 'he_normal'

        x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

        x = convolutional_block(self, x, filters, 3, [1,2,4], initializer)
        #x = convolutional_block(self, inputs, filters, 3, [1,2,4], initializer)

        x = Bidirectional(LSTM(units=filters, return_sequences=True, activation='tanh'))(x)
        x = Bidirectional(LSTM(units=filters//2, activation='tanh'))(x)

        x = Dense(units=128, activation='selu')(x)

        def sigmoid(x):
            return -12.5 + ((6.5-(-12.5)) * K.sigmoid(x))

        output_network = Dense(units=self.trajectory_length, activation=Activation(sigmoid))(x)#'sigmoid')(x)

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
        return f'segment_diffusion_coefficient'

    def plot_bias(self):
        trajectories = self.simulate_trajectories(VALIDATION_SET_SIZE_PER_EPOCH)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_bias(ground_truth, predicted, symbol='d')

    def plot_predicted_and_ground_truth_distribution(self):
        trajectories = self.simulate_trajectories(VALIDATION_SET_SIZE_PER_EPOCH)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_predicted_and_ground_truth_distribution(ground_truth, predicted)

    def plot_predicted_and_ground_truth_histogram(self):
        trajectories = self.simulate_trajectories(VALIDATION_SET_SIZE_PER_EPOCH)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        predicted = self.predict(trajectories).flatten()

        plot_predicted_and_ground_truth_histogram(ground_truth, predicted)

    def adjust_trajectories_for_training_and_validation(self, trajectories):
        new_trajectories = []
        for trajectory_index in range(len(trajectories)):

            for i in range(0, trajectories[trajectory_index].length, self.trajectory_length):
                new_sub_trajectory = trajectories[trajectory_index].build_noisy_subtrajectory_from_range(i, i+self.trajectory_length)

                if new_sub_trajectory.length == self.trajectory_length:
                    new_trajectories.append(new_sub_trajectory)
                    assert new_sub_trajectory.length == self.trajectory_length
                    #assert len(np.unique(new_sub_trajectory.info['D'])) == 1

            """
            cps = detect_change_points(trajectories[trajectory_index].info['D'])

            if len(cps) == 0:
                sub_trajectories = [trajectories[trajectory_index]]
            else:
                i = 0
                for cp in cps:
                    sub_trajectories.append(trajectories[trajectory_index].build_noisy_subtrajectory_from_range(i, i+cp))
                    i = cp

            for sub_trajectory in sub_trajectories:
                for i in range(0, sub_trajectory.length, self.trajectory_length):
                    new_sub_trajectory = sub_trajectory.build_noisy_subtrajectory_from_range(i, i+self.trajectory_length)

                    if new_sub_trajectory.length == self.trajectory_length:
                        new_trajectories.append(new_sub_trajectory)
                        assert new_sub_trajectory.length == self.trajectory_length
                        #assert len(np.unique(new_sub_trajectory.info['D'])) == 1
                    else:
                        print(new_sub_trajectory.length)
            """
            """
            initial_index = np.random.randint(trajectories[trajectory_index].length-self.trajectory_length)
            trajectories[trajectory_index] = trajectories[trajectory_index].build_noisy_subtrajectory_from_range(initial_index, initial_index+self.trajectory_length)
            assert trajectories[trajectory_index].length == self.trajectory_length
            """


        return new_trajectories
    """
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
    """