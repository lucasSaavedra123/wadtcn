import numpy as np
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate, Add, Multiply
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import to_categorical

from .PredictiveModel import PredictiveModel
from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionBrownian, FractionalBrownianMotionSuperDiffusive
from .model_utils import transform_trajectories_into_displacements, convolutional_block, WaveNetEncoder

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
            'training_set_size': 100000,
            'validation_set_size': 12500,
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 3,
            'with_early_stopping': True,
            'dropout_rate': 0
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
        # Net filters and kernels
        initializer = 'he_normal'
        filters = 64
        x1_kernel = 4
        x2_kernel = 2
        x3_kernel = 3
        x4_kernel = 10
        x5_kernel = 20

        dilation_depth = 8

        inputs = Input(shape=(self.trajectory_length-1, 2))

        x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

        x1 = convolutional_block(self, x, filters, x1_kernel, [1,2,4], initializer)
        x2 = convolutional_block(self, x, filters, x2_kernel, [1,2,4], initializer)
        x3 = convolutional_block(self, x, filters, x3_kernel, [1,2,4], initializer)
        x4 = convolutional_block(self, x, filters, x4_kernel, [1,4,8], initializer)

        x5 = Conv1D(filters=filters, kernel_size=x5_kernel, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x5 = BatchNormalization()(x5)

        x = concatenate(inputs=[x1, x2, x3, x4, x5])

        x = GlobalMaxPooling1D()(x)

        dense_1 = Dense(units=512, activation='relu')(x)
        dense_2 = Dense(units=128, activation='relu')(dense_1)
        output_network = Dense(units=self.number_of_models_involved, activation='softmax')(dense_2)

        self.architecture = Model(inputs=inputs, outputs=output_network)

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
        Y_as_vectors = np.empty((len(trajectories), self.number_of_models_involved))

        for index, trajectory in enumerate(trajectories):
            Y_as_vectors[index, :] = to_categorical(self.model_to_label(trajectory.model_category), num_classes=self.number_of_models_involved)

        return Y_as_vectors

    def model_to_label(self, model):
        return self.models_involved_in_predictive_model.index(model.__class__)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_displacements(self, trajectories)

    def type_name(self):
        return f"wavenet_tcn_fbm_model_classifier"
