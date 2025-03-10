import os
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import glob
from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
import pandas as pd
from Trajectory import Trajectory
from keras.callbacks import EarlyStopping
from tensorflow import device
import keras.backend as K
from utils import break_point_detection_with_stepfinder
from andi_datasets.datasets_challenge import _defaults_andi2


class WavenetTCNSingleLevelDiffusionCoefficientPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':999}

    @classmethod
    def selected_hyperparameters(self):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':999}

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [32, 64, 128, 256],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories), verbose=0)

    def transform_trajectories_to_output(self, trajectories):
        Y = transform_trajectories_to_single_level_diffusion_coefficient(self, trajectories)
        Y = np.log10(Y)
        return  Y

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_displacements(self, trajectories, normalize=False)
        X = np.hstack([np.zeros((len(trajectories), 1, 2)), X])
        return X

    def build_network(self, hp=None):
        number_of_features = 2
        wavenet_filters = 32
        dff = 320
        number_of_passes = 2

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

        x = Transformer(2,4,wavenet_filters*5,320)(x)

        # def custom_tanh_1(x):
        #     return (K.tanh(x)+1)/2

        # alpha_regression = Conv1D(filters=wavenet_filters*5, kernel_size=3, padding='causal', activation='relu', kernel_initializer=initializer)(x)
        # alpha_regression = TimeDistributed(Dense(units=1, activation=custom_tanh_1), name='alpha_regression_output')(alpha_regression)

        def custom_tanh_2(x):
            return (K.tanh(x)*9*1.10)-3

        d_regression = Dense(units=1, activation=custom_tanh_2, name='d_regression_output')(x)

        self.architecture = Model(inputs=inputs, outputs=d_regression)

        if hp is not None:
            hyperparameter_search_range = self.__class__.default_hyperparameters_analysis()
            optimizer = Adam(
                learning_rate=hp.Choice('learning_rate', values=hyperparameter_search_range['lr']),
                epsilon=hp.Choice('epsilon', values=hyperparameter_search_range['epsilon']),
                amsgrad=hp.Choice('amsgrad', values=hyperparameter_search_range['amsgrad'])
            )
        else:
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
        return 'wavenet_single_level_inference_diffusion_coefficient'

    @property
    def dataset_type(self):
        return 'regression'

    def prepare_dataset(self, set_size, files):
        trajectories = np.random.choice(files,size=set_size, replace=False).tolist()
        for i in range(len(trajectories)):
            df = pd.read_csv(trajectories[i])
            df = df.sort_values('t', ascending=True)

            sigma = np.random.uniform(0,2)

            trajectories[i] = Trajectory(
                x=df['x'].tolist(),
                y=df['y'].tolist(),
                noise_x=np.random.randn(len(df)) * sigma,
                noise_y=np.random.randn(len(df)) * sigma,
                info={
                    'd_t':df['d_t'].tolist(),
                    'alpha_t':df['alpha_t'].tolist(),
                    'state_t':df['state_t'].tolist()
                }
            )

        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}"

    def plot_single_level_prediction(self, limit=10, sigma=0):
        trajectories = self.simulator().simulate_phenomenological_trajectories_for_regression_training(VALIDATION_SET_SIZE_PER_EPOCH,self.trajectory_length,None,True,'val', ignore_boundary_effects=True)
 
        for t in trajectories:
            t.x = (np.array(t.x) + np.random.randn(t.length)*sigma).tolist()
            t.y = (np.array(t.y) + np.random.randn(t.length)*sigma).tolist()
        
        np.random.shuffle(trajectories)
        result = self.predict(trajectories[:limit])
        idxs = np.arange(0,limit, 1)

        for i in idxs:
            ti = trajectories[i]

            fig, ax = plt.subplots(1,2)
            ax[0].plot(ti.get_noisy_x(), ti.get_noisy_y())
            ax[1].set_title('D')
            ax[1].plot(np.log10(ti.info['d_t']), color='black')
            ax[1].scatter(range(len(result[i, :])), result[i, :], color='red')

            for bkp in break_point_detection_with_stepfinder(np.log10(ti.info['d_t']), tresH=D_ACCEPTANCE_THRESHOLD):
                ax[1].axvline(bkp, color='blue')

            ax[1].set_ylim([-12,6])

            plt.show()
