from keras.layers import Dense, Input, Average, TimeDistributed
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
from keras.callbacks import EarlyStopping
from tensorflow import device, config


class WavenetTCNHurstExponentSingleLevelPredicter(PredictiveModel):
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

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories))

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_single_level_hurst_exponent(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories, normalize=True)

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
            lr=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    @property
    def type_name(self):
        return 'wavenet_single_level_hurst_exponent'

    def prepare_dataset(self, set_size, file_label='', get_from_cache=False):
        trajectories = self.simulator().simulate_phenomenological_trajectories(set_size, self.trajectory_length, self.trajectory_time, get_from_cache=get_from_cache, file_label=file_label)
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

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

    def fit(self):
        if not self.trained:
            self.build_network()
            real_epochs = self.hyperparameters['epochs']
        else:
            real_epochs = self.hyperparameters['epochs'] - len(self.history_training_info['loss'])

        self.architecture.summary()

        if self.early_stopping:
            callbacks = [EarlyStopping(
                monitor="val_loss",
                min_delta=1e-3,
                patience=5,
                verbose=1,
                mode="min")]
        else:
            callbacks = []

        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        X_train, Y_train = self.prepare_dataset(12_500, file_label='train', get_from_cache=True)
        X_val, Y_val = self.prepare_dataset(12_500, file_label='val', get_from_cache=True)

        with device(device_name):
            history_training_info = self.architecture.fit(
                X_train, Y_train,
                epochs=real_epochs,
                callbacks=callbacks,
                batch_size=self.hyperparameters['batch_size'],
                validation_data=[X_val, Y_val],
                shuffle=True
            ).history

        if self.trained:
            for dict_key in history_training_info:
                self.history_training_info[dict_key] += history_training_info[dict_key]
        else:
            self.history_training_info = history_training_info
            self.trained = True

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}"