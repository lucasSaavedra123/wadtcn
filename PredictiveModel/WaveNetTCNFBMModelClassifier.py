import numpy as np
from tensorflow.keras.optimizers.legacy import Adam

from .PredictiveModel import PredictiveModel
from TheoreticalModels import FBM_MODELS
from .model_utils import transform_trajectories_into_displacements, build_more_complex_wavenet_tcn_classifier_for, transform_trajectories_to_categorical_vector


class WaveNetTCNFBMModelClassifier(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return FBM_MODELS

    @classmethod
    def selected_hyperparameters(cls):
        return {
            'batch_size': 8,
            'amsgrad': True,
            'epsilon': 1e-8,
            'epochs': 100,
            'lr': 0.0001
        }

    def default_hyperparameters(self):
        return {
            'batch_size': 8,
            'amsgrad': True,
            'epsilon': 1e-8,
            'epochs': 100,
            'lr': 0.0001
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 16, 32, 64],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    def build_network(self):
        build_more_complex_wavenet_tcn_classifier_for(self)

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

    @property
    def type_name(self):
        return f"wavenet_tcn_fbm_model_classifier"
