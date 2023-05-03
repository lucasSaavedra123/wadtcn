import numpy as np
from keras.layers import Dense, Input, LSTM
from keras.models import Model

from .PredictiveModel import PredictiveModel

from .model_utils import transform_trajectories_into_displacements, transform_trajectories_to_anomalous_exponent

class LSTMAnomalousExponentPredicter(PredictiveModel):
    def default_hyperparameters(self):
        return {
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-6,
            'epochs': 100,
            'lr': 0.01
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 32, 128, 256, 512],
            'epsilon': [1e-6, 1e-7, 1e-8],
        }


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
        outputs = Dense(1)(x)

        self.architecture = Model(inputs=inputs, outputs=outputs)                             

        self.architecture.compile(optimizer='adam', loss="mse", metrics=["mae"])

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        Y_predicted = self.architecture.predict(X)
        Y_predicted = np.argmax(Y_predicted, axis=-1)
        return Y_predicted

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_anomalous_exponent(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_displacements(self, trajectories)

    @property
    def type_name(self):
        return f"lstm_anomalous_exponent_predicter"
