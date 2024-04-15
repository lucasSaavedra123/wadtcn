from keras.layers import Dense, Input, Average, Conv1D, TimeDistributed
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError


from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *


class WavenetTCNDiffusionCoefficientSingleLevelPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        hyperparameters = {'lr': 0.0002, 'batch_size': 512, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 100}
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
        #Check this later to be between 0 and 1
        d = transform_trajectories_to_single_level_diffusion_coefficient(self, trajectories)
        d[d==0] = np.finfo(float).eps
        return np.log10(d)

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)
        return X

    def build_network(self):
        number_of_features = 2
        inputs = Input(shape=(self.trajectory_length, number_of_features))
        wavenet_filters = 16
        dilation_depth = 8
        initializer = 'he_normal'

        x = WaveNetEncoder(wavenet_filters, dilation_depth, initializer=initializer)(inputs)
        unet_1 = Unet((self.trajectory_length, wavenet_filters), '1d', 2, unet_index=1, skip_last_block=True)(x)
        unet_2 = Unet((self.trajectory_length, wavenet_filters), '1d', 3, unet_index=2, skip_last_block=True)(x)
        unet_3 = Unet((self.trajectory_length, wavenet_filters), '1d', 4, unet_index=3, skip_last_block=True)(x)
        unet_4 = Unet((self.trajectory_length, wavenet_filters), '1d', 9, unet_index=4, skip_last_block=True)(x)            

        x = concatenate([unet_1, unet_2, unet_3, unet_4])
        #output_network = Conv1D(1, 3, 1, padding='same', activation='sigmoid')(x)
        output_network = TimeDistributed(Dense(units=1, activation='sigmoid'))(x)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
            learning_rate=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )
        """
        from tensorflow import reduce_mean, square, reshape, abs
        def custom_mse(y_true, y_pred):
            y_true = reshape(y_true, [-1])
            y_pred = reshape(y_pred, [-1])
            return reduce_mean(square(y_true - y_pred))

        def custom_mae(y_true, y_pred):
            y_true = reshape(y_true, [-1])
            y_pred = reshape(y_pred, [-1])
            return reduce_mean(abs(y_true - y_pred))

        self.architecture.compile(optimizer=optimizer, loss=custom_mse, metrics=[custom_mse, custom_mae])
        """
        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
    @property
    def type_name(self):
        return 'wavenet_single_level_diffusion_coefficient'

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

    def change_network_length_to(self, new_length):
        self.trajectory_length = new_length
        old_architecture = self.architecture
        self.build_network()

        for i, _ in enumerate(old_architecture.layers):
            if i == 0:
                continue
            else:
                if 'unet' in old_architecture.layers[i].name:
                    for j, _ in enumerate(old_architecture.layers[i].layers):
                        if j == 0 or old_architecture.layers[i].layers[j].name == 'LeakyReLU':
                            continue
                        else:
                            self.architecture.layers[i].layers[j].set_weights(old_architecture.layers[i].layers[j].get_weights())
                else:
                    self.architecture.layers[i].set_weights(old_architecture.layers[i].get_weights())
