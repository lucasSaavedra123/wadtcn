import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Dense, Input, LSTM, Conv1D, Bidirectional
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from TheoreticalModels.AnnealedTransientTimeMotion import AnnealedTransientTimeMotion
from TheoreticalModels.ContinuousTimeRandomWalk import ContinuousTimeRandomWalk
from TheoreticalModels.LevyWalk import LevyWalk
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionBrownian, FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionSuperDiffusive
from TheoreticalModels.ScaledBrownianMotion import ScaledBrownianMotionBrownian, ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionSuperDiffusive
from .PredictiveModel import PredictiveModel
from .model_utils import transform_trajectories_into_displacements

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
            'epochs': 50
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
        return [self.hyperparameters[self.extra_parameters["model"]]]

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

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length, 1))
        x = Conv1D(64, kernel_size=5, padding='causal', activation='relu')(inputs)
        x = Bidirectional(LSTM(units=64, return_sequences=True, activation='tanh'))(x)
        x = Bidirectional(LSTM(units=32, activation='tanh'))(x)
        x = Dense(units=128, activation='tanh')(x)
        output_network = Dense(units=1, activation='tanh')(x)

        keras_model = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
            lr=self.hyperparameters[self.extra_parameters["model"].STRING_LABEL]['lr'],
            epsilon=self.hyperparameters[self.extra_parameters["model"].STRING_LABEL]['epsilon'],
            amsgrad=self.hyperparameters[self.extra_parameters["model"].STRING_LABEL]['amsgrad']
        )

        keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

        self.architecture = keras_model

    def plot_bias(self, trajectories=None):
        trajectories = self.simulator().simulate_trajectories(self.hyperparameters['validation_set_size'], self.trajectory_length, self.models_involved_in_predictive_model, self.trajectory_time)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()

        difference = Y_predicted - ground_truth

        sns.kdeplot(difference.flatten(), color='blue', shade=True)
        plt.rcParams.update({'font.size': 15})
        plt.ylabel('Frequency', fontsize=15)
        plt.xlabel(r'$\alpha _{P} - \alpha _{GT}$', fontsize=15)
        plt.grid()
        plt.show()

    def plot_predicted_and_ground_truth_distribution(self, trajectories=None):
        trajectories = self.simulator().simulate_trajectories(self.hyperparameters['validation_set_size'], self.trajectory_length, self.models_involved_in_predictive_model, self.trajectory_time)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()

        sns.kdeplot(ground_truth, color='green', shade=True)
        sns.kdeplot(Y_predicted, color='red', shade=True)
        plt.rcParams.update({'font.size': 15})
        plt.ylabel('Frequency', fontsize=15)
        plt.xlabel(r'Values', fontsize=15)
        plt.grid()
        plt.show()

    def plot_predicted_and_ground_truth_histogram(self, trajectories=None):
        trajectories = self.simulator().simulate_trajectories(self.hyperparameters['validation_set_size'], self.trajectory_length, self.models_involved_in_predictive_model, self.trajectory_time)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten()
        Y_predicted = self.predict(trajectories).flatten()

        plt.hist2d(ground_truth, Y_predicted, bins=50, range=[[0, 1], [0, 1]])
        plt.rcParams.update({'font.size': 15})
        plt.ylabel('Predicted', fontsize=15)
        plt.xlabel('Ground Truth', fontsize=15)
        plt.grid()
        plt.show()
