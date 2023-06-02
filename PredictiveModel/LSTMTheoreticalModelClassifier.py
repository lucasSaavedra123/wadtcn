import numpy as np
from keras.layers import Dense, Input, LSTM
from keras.models import Model

from .PredictiveModel import PredictiveModel
from TheoreticalModels import ALL_MODELS, ANDI_MODELS

from .model_utils import transform_trajectories_into_displacements, transform_trajectories_to_categorical_vector

class LSTMTheoreticalModelClassifier(PredictiveModel):
    @classmethod
    def selected_hyperparameters(self):
        return {
            'batch_size': 32,
            'epochs': 100,
        }

    def default_hyperparameters(self):
        return {
            'batch_size': 32,
            'epochs': 100,
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        pass

    """
    This network comes from paper:

    Aykut Argun, Giovanni Volpe, Stefano Bo

    Classification, Inference, and Segmentation of anomalous
    diffusion with recurrent neural networks

    Original code: https://github.com/argunaykut/randi/blob/main/classification_train_network.ipynb
    """
    def build_network(self):
        inputs = Input((self.trajectory_length-1, 2))

        x = LSTM(250, return_sequences=True)(inputs)
        x = LSTM(50)(x)
        x = Dense(20)(x)
        outputs = Dense(len(self.models_involved_in_predictive_model), activation="softmax")(x)

        self.architecture = Model(inputs=inputs, outputs=outputs)                             

        self.architecture.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        Y_predicted = self.architecture.predict(X)
        Y_predicted = np.argmax(Y_predicted, axis=-1)
        return Y_predicted

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_displacements(self, trajectories)

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_categorical_vector(self, trajectories)

    @property
    def type_name(self):
        return f"lstm_theoretical_model_classifier"
