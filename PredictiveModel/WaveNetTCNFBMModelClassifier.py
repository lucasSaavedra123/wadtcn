import numpy as np
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate, Add, Multiply
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical

from .PredictiveModel import PredictiveModel
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionBrownian, FractionalBrownianMotionSuperDiffusive
from .model_utils import transform_trajectories_into_displacements, build_wavenet_tcn_classifier_for, transform_trajectories_to_categorical_vector

class WaveNetTCNFBMModelClassifier(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return [FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionBrownian, FractionalBrownianMotionSuperDiffusive]

    @property
    def number_of_models_involved(self):
        return len(self.models_involved_in_predictive_model)

    #These will be updated after hyperparameter search
    def default_hyperparameters(self):
        return {
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 100,
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 32, 128, 256, 512],
            'epsilon': [1e-6, 1e-7, 1e-8],
        }

    def build_network(self):
        build_wavenet_tcn_classifier_for(self)

        optimizer = Adam(lr=self.hyperparameters['lr'],
                         amsgrad=self.hyperparameters['amsgrad'],
                         epsilon=self.hyperparameters['epsilon'])

        self.architecture.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        Y_predicted = self.architecture.predict(X)
        Y_predicted = np.argmax(Y_predicted, axis=-1)
        return Y_predicted

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_categorical_vector(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_displacements(self, trajectories)

    def type_name(self):
        return f"wavenet_tcn_fbm_model_classifier"
