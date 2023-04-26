import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Input, LSTM, Conv1D, Bidirectional, BatchNormalization, GlobalMaxPooling1D, Add, Multiply
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from TheoreticalModels.AnnealedTransientTimeMotion import AnnealedTransientTimeMotion
from TheoreticalModels.ContinuousTimeRandomWalk import ContinuousTimeRandomWalk
from TheoreticalModels.LevyWalk import LevyWalk
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionBrownian, FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionSuperDiffusive
from TheoreticalModels.ScaledBrownianMotion import ScaledBrownianMotionBrownian, ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionSuperDiffusive
from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_raw_trajectories, transform_trajectories_into_displacements

class HurstExponentModel(PredictiveModel):

    #These will be updated after hyperparameter search
    def default_hyperparameters(self):
        return {
            'training_set_size': 100000,
            'validation_set_size': 12500,
            'with_early_stopping': False,
            'fbm_sub': {
                'lr': 0.0001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-6,
                'model': FractionalBrownianMotionSubDiffusive
            },
            'fbm_brownian': {
                'lr': 0.0001,
                'batch_size': 16,
                'amsgrad': True,
                'epsilon': 1e-7,
                'model': FractionalBrownianMotionBrownian
            },
            'fbm_sup': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': FractionalBrownianMotionSuperDiffusive
            },
            'sbm_sub': {
                'lr': 0.0001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-6,
                'model': ScaledBrownianMotionSubDiffusive
            },
            'sbm_brownian': {
                'lr': 0.0001,
                'batch_size': 16,
                'amsgrad': True,
                'epsilon': 1e-7,
                'model': ScaledBrownianMotionBrownian
            },
            'sbm_sup': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': ScaledBrownianMotionSuperDiffusive
            },
            'lw': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': LevyWalk
            },
            'ctrw': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': ContinuousTimeRandomWalk
            },
            'attm': {
                'lr': 0.001,
                'batch_size': 64,
                'amsgrad': False,
                'epsilon': 1e-7,
                'model': AnnealedTransientTimeMotion
            },
            'epochs': 5
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [16, 32, 64, 128],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return [self.hyperparameters[self.extra_parameters["model"]]['model']]

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        return self.architecture.predict(X)

    def transform_trajectories_to_output(self, trajectories):
        Y = np.empty((len(trajectories), 1))

        for index, trajectory in enumerate(trajectories):
            Y[index, 0] = trajectory.hurst_exponent()

        return Y

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_displacements(self, trajectories)

    def conv_bloc(self, original_x, filters, kernel_size, dilation_rates, initializer):
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu', kernel_initializer=initializer, dilation_rate=dilation_rates[0])(original_x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rates[1], padding='causal', activation='relu', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rates[2], padding='causal', activation='relu', kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)

        x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_initializer=initializer)(original_x)
        x_skip = BatchNormalization()(x_skip)

        x = Add()([x, x_skip])

        return x

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length-1, 2))
        dilation_depth = 8
        initializer = 'he_normal'
        filters = 64

        wavenet_dilations = [2**i for i in range(dilation_depth)]
        conv_1d_tanh = [Conv1D(filters, kernel_size=3, dilation_rate=dilation, padding='causal', activation='tanh') for dilation in wavenet_dilations]
        conv_1d_sigm = [Conv1D(filters, kernel_size=3, dilation_rate=dilation, padding='causal', activation='sigmoid') for dilation in wavenet_dilations]

        x = Conv1D(filters, 3, padding='causal')(inputs)

        layers_to_add = [x]

        for i in range(dilation_depth):
            tanh_out = conv_1d_tanh[i](x)
            sigm_out = conv_1d_sigm[i](x)

            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters, 1, padding='causal')(x)

            layers_to_add.append(x)

        x = Add()(layers_to_add)
        x = BatchNormalization()(x)

        x = self.conv_bloc(x, filters, 5, [1,2,4], initializer)

        x = Bidirectional(LSTM(units=filters, return_sequences=True, activation='tanh'))(x)
        x = Bidirectional(LSTM(units=filters//2, activation='tanh'))(x)
        x = Dense(units=128, activation='selu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
            lr=self.hyperparameters[self.extra_parameters["model"]]['lr'],
            epsilon=self.hyperparameters[self.extra_parameters["model"]]['epsilon'],
            amsgrad=self.hyperparameters[self.extra_parameters["model"]]['amsgrad']
        )

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    def plot_bias(self):
        trajectories = self.simulator().simulate_trajectories_by_model(self.hyperparameters['validation_set_size'], self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()

        difference = Y_predicted - ground_truth

        sns.kdeplot(difference.flatten(), color='blue', fill=True)
        plt.rcParams.update({'font.size': 15})
        plt.ylabel('Frequency', fontsize=15)
        plt.xlabel(r'$\alpha _{P} - \alpha _{GT}$', fontsize=15)
        plt.grid()
        plt.show()

    def plot_predicted_and_ground_truth_distribution(self):
        trajectories = self.simulator().simulate_trajectories_by_model(self.hyperparameters['validation_set_size'], self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()

        sns.kdeplot(ground_truth, color='green', fill=True)
        sns.kdeplot(Y_predicted, color='red', fill=True)
        plt.rcParams.update({'font.size': 15})
        plt.ylabel('Frequency', fontsize=15)
        plt.xlabel(r'Values', fontsize=15)
        plt.grid()
        plt.show()

    def plot_predicted_and_ground_truth_histogram(self):
        trajectories = self.simulator().simulate_trajectories_by_model(self.hyperparameters['validation_set_size'], self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()

        plt.hist2d(ground_truth, Y_predicted, bins=50, range=[[0, 2], [0, 2]])
        plt.rcParams.update({'font.size': 15})
        plt.ylabel('Predicted', fontsize=15)
        plt.xlabel('Ground Truth', fontsize=15)
        plt.grid()
        plt.show()

    @property
    def type_name(self):
        return 'hurst_exponent'