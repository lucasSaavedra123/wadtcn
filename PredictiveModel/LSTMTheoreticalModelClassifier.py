import numpy as np
from keras.layers import Dense, Input, LSTM
from keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
from tensorflow import device, config

from .PredictiveModel import PredictiveModel
from TheoreticalModels import ALL_MODELS, ANDI_MODELS

from CONSTANTS import *
from .model_utils import transform_trajectories_into_displacements, transform_trajectories_to_categorical_vector

class LSTMTheoreticalModelClassifier(PredictiveModel):
    @classmethod
    def selected_hyperparameters(self):
        return {}

    def default_hyperparameters(self):
        return {}

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
        model_classification = Sequential()

        model_classification.add(LSTM(250,                         # first layer: LSTM of dimension 250
                                return_sequences=True,            # return sequences for the second LSTM layer            
                                recurrent_dropout=0.2,            # recurrent dropout for preventing overtraining
                                input_shape=(None, self.block_size)))  # input shape

        model_classification.add(LSTM(50,                          # second layer: LSTM of dimension 50
                                dropout=0,
                                recurrent_dropout=0.2))

        model_classification.add(Dense(20))                        # dense layer 

        model_classification.add(Dense(5,                          # output layer, each node for predicting prob. for each model
                                activation="softmax",))                              

        model_classification.compile(optimizer='adam',
                        loss="categorical_crossentropy",
                        metrics=["categorical_accuracy"])

        self.architecture = model_classification

    def predict(self, trajectories):
        X = self.transform_trajectories_to_input(trajectories)
        Y_predicted = self.architecture.predict(X)
        Y_predicted = np.argmax(Y_predicted, axis=-1)
        return Y_predicted

    def transform_trajectories_to_input(self, trajectories):
        X = np.zeros((len(trajectories), (self.trajectory_length-1)//self.block_size, self.block_size))

        for trajectory_index, trajectory in enumerate(trajectories):
            disp_x = np.diff(trajectory.get_noisy_x())
            disp_y = np.diff(trajectory.get_noisy_y())
        
            raw = []

            for i in range(self.trajectory_length-1):
                raw.append(disp_x[i])
                raw.append(disp_y[i])

            for number_of_block in range(X.shape[1]):
                for position_in_block in range(X.shape[2]):
                    X[trajectory_index, number_of_block, position_in_block] = raw[(number_of_block * position_in_block) + position_in_block]

        return X

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_categorical_vector(self, trajectories)

    @property
    def type_name(self):
        return f"lstm_theoretical_model_classifier"

    def fit(self):
        self.build_network()

        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        with device(device_name):
            batch_sizes = [32, 128, 512, 2048]
            dataset_used = [1, 4, 5, 20]
            number_epochs = [5, 4, 3, 2]
            N = 100000

            for batch in range(len(batch_sizes)):    
                for repeat in range(dataset_used[batch]):
                    x, label = self.prepare_dataset(N)
                    self.architecture.fit(
                        x,
                        label, 
                        epochs=number_epochs[batch], 
                        batch_size=batch_sizes[batch],
                        validation_split=0.1,
                        shuffle=True
                    )

        self.trained = True
    
    @property
    def block_size(self):
        if self.trajectory_length == 25 or self.trajectory_length == 65:
            return 4
        else:
            return 8
