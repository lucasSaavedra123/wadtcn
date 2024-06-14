import os
import numpy as np
from keras.layers import Dense, Input, TimeDistributed
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import glob
from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import confusion_matrix, f1_score
from Trajectory import Trajectory
from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
import pandas as pd
from keras.callbacks import EarlyStopping
from tensorflow import device, config
import keras.backend as K

class WavenetTCNMultiTaskSingleLevelPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':100}

    @classmethod
    def selected_hyperparameters(self):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':100}

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [32, 64, 128, 256, 512, 1024],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    #@property
    #def models_involved_in_predictive_model(self):
    #    return ['trap', 'confined', 'free', 'directed']

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories), verbose=0)

    def transform_trajectories_to_output(self, trajectories):
        Y1 = transform_trajectories_to_single_level_hurst_exponent(self, trajectories)

        Y2 = transform_trajectories_to_single_level_diffusion_coefficient(self, trajectories)
        Y2[Y2==0] = 1e-12
        Y2 = np.log10(Y2)
        #Y2 = (Y2 + 12)/18

        return  Y1, Y2

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories)
        return X

    def build_network(self):
        number_of_features = 2
        wavenet_filters = 32
        dilation_depth = 8
        initializer = 'he_normal'
        x1_kernel = 4
        x2_kernel = 2
        x3_kernel = 3
        x4_kernel = 10
        x5_kernel = 20

        dilation_depth = 8

        inputs = Input(shape=(None, number_of_features))

        x = WaveNetEncoder(wavenet_filters, dilation_depth, initializer=initializer)(inputs)

        x1 = convolutional_block(self, x, wavenet_filters, x1_kernel, [1,2,4], initializer)
        x2 = convolutional_block(self, x, wavenet_filters, x2_kernel, [1,2,4], initializer)
        x3 = convolutional_block(self, x, wavenet_filters, x3_kernel, [1,2,4], initializer)
        x4 = convolutional_block(self, x, wavenet_filters, x4_kernel, [1,4,8], initializer)

        x5 = Conv1D(filters=wavenet_filters, kernel_size=x5_kernel, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x5 = BatchNormalization()(x5)

        x = concatenate(inputs=[x1, x2, x3, x4, x5])
        x_1 = x
        #Following code is similar to Requena, 2023.
        for _ in range(2):
            x = EncoderLayer(d_model=wavenet_filters*5, num_heads=4, dff=320, dropout_rate=0.1)(x)
        x = Add()([x_1, x])

        x = LayerNormalization()(x)
        x_1 = x
        x = FeedForward(wavenet_filters*5, 320, 0.1)(x)
        x = Add()([x_1, x])
        x = LayerNormalization()(x)

        x = FeedForward(wavenet_filters*5, 320, 0.1)(x)

        # def custom_tanh_1(x):
        #     return (K.tanh(x)+1)/2

        # alpha_regression = Conv1D(filters=wavenet_filters*5, kernel_size=3, padding='causal', activation='relu', kernel_initializer=initializer)(x)
        # alpha_regression = TimeDistributed(Dense(units=1, activation=custom_tanh_1), name='alpha_regression_output')(alpha_regression)

        def custom_tanh_2(x):
            return (K.tanh(x)*9*1.1)-3

        d_regression = Conv1D(filters=wavenet_filters*5, kernel_size=1, padding='causal', activation='relu', kernel_initializer=initializer)(x)
        d_regression = Dense(units=1, activation=custom_tanh_2, name='d_regression_output')(d_regression)

        # self.architecture = Model(inputs=inputs, outputs=[alpha_regression, d_regression])
        self.architecture = Model(inputs=inputs, outputs=d_regression)
        optimizer = Adam(
            learning_rate=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )
        """
        loss_parameter = {
            'alpha_regression_output': 'mse',
            'd_regression_output': 'mse'
        }

        metrics_parameter = {
            'alpha_regression_output': 'mae',
            'd_regression_output': 'mae'
        }

        self.architecture.compile(optimizer=optimizer, loss=loss_parameter, metrics=metrics_parameter)
        """
        self.architecture.compile(optimizer=optimizer, loss='mae' , metrics='mae')
    @property
    def type_name(self):
        return 'wavenet_single_level_model'

    def prepare_dataset(self, set_size, file_label='', get_from_cache=False):
        trajectories = self.simulator().simulate_phenomenological_trajectories(set_size, self.trajectory_length, self.trajectory_time, get_from_cache=get_from_cache, file_label=file_label, enable_parallelism=True, type_of_simulation='models_phenom')
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

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

        X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH, file_label='val', get_from_cache=True)
        Y1_val = Y_val#Y_val[0]
        #Y2_val = Y_val[1]

        number_of_training_trajectories = len(glob.glob('./2ndAndiTrajectories/*_X.npy'))

        def custom_prepare_dataset(batch_size):            
            trajectories_ids = np.random.randint(number_of_training_trajectories, size=batch_size)

            X, Y1, Y2 = [], [], []

            for trajectory_id in trajectories_ids:
                X.append(np.load(os.path.join('./2ndAndiTrajectories', f'{trajectory_id}_X.npy')))
                Y1.append(np.load(os.path.join('./2ndAndiTrajectories', f'{trajectory_id}_Y1.npy')))
                Y2.append(np.load(os.path.join('./2ndAndiTrajectories', f'{trajectory_id}_Y2.npy')))

            X = np.concatenate(X)
            Y1 = np.concatenate(Y1)
            Y2 = np.concatenate(Y2)

            return X, Y2#[Y1, Y2]

        with device(device_name):
            history_training_info = self.architecture.fit(
                TrackGenerator(TRAINING_SET_SIZE_PER_EPOCH//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], custom_prepare_dataset),
                epochs=real_epochs,
                callbacks=callbacks,
                batch_size=self.hyperparameters['batch_size'],
                validation_data=[X_val, Y1_val],#[Y1_val, Y2_val]],
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

    def plot_single_level_prediction(self, limit=10):
        trajectories = self.simulator().simulate_phenomenological_trajectories(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, get_from_cache=True, file_label='val')
        np.random.shuffle(trajectories)
        result = self.predict(trajectories[:limit])
        idxs = np.arange(0,limit, 1)

        for i in idxs:
            fig, ax = plt.subplots(2,1)
            ti = trajectories[i]

            ax[0].set_title('Alpha')
            ax[0].plot(ti.info['alpha_t'], color='black')
            #ax[0].plot(result[i, :]*2, color='red')
            ax[0].set_ylim([0,2])

            ax[1].set_title('D')
            ax[1].plot(np.log10(ti.info['d_t']), color='black')
            ax[1].plot(result[i, :], color='red')
            ax[1].set_ylim([-12,6])

            plt.show()
