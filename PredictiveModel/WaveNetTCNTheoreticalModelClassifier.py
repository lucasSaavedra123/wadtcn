import numpy as np
from tensorflow.keras.optimizers.legacy import Adam

from .PredictiveModel import PredictiveModel
from TheoreticalModels import ANDI_MODELS, ALL_MODELS
from .model_utils import transform_trajectories_into_displacements, build_wavenet_tcn_classifier_for, transform_trajectories_to_categorical_vector, build_wavenet_tcn_classifier_from_encoder_for


class WaveNetTCNTheoreticalModelClassifier(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return ANDI_MODELS if self.simulator.STRING_LABEL == 'andi' else ALL_MODELS

    @classmethod
    def selected_hyperparameters(cls):
        return {
            'batch_size': 64,
            'amsgrad': True,
            'epsilon': 1e-08,
            'epochs': 100,
            'lr': 0.001
        }

    def default_hyperparameters(self):
        return {
            'batch_size': 64,
            'amsgrad': True,
            'epsilon': 1e-08,
            'epochs': 100,
            'lr': 0.001
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
        if self.wadnet_tcn_encoder is None:
            build_wavenet_tcn_classifier_for(self)
        else:
            build_wavenet_tcn_classifier_from_encoder_for(self, 320)

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
        X = transform_trajectories_into_displacements(self, trajectories)
        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X)
        return X

    @property
    def type_name(self):
        return "wavenet_tcn_theoretical_model_classifier"