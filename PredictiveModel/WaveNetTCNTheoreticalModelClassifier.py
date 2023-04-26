import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate, Add, Multiply
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
#from tensorflow.keras.metrics import F1Score
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.utils import to_categorical

from .PredictiveModel import PredictiveModel
from TheoreticalModels import ANDI_MODELS, ALL_MODELS
from .model_utils import transform_trajectories_into_displacements, WaveNetEncoder


class WaveNetTCNTheoreticalModelClassifier(PredictiveModel):
    @property
    def models_involved_in_predictive_model(self):
        return ANDI_MODELS if self.simulator().STRING_LABEL == 'andi' else ALL_MODELS

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
            'epochs': 100,
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

        x1 = self.conv_bloc(x, filters, x1_kernel, [1,2,4], initializer)
        x2 = self.conv_bloc(x, filters, x2_kernel, [1,2,4], initializer)
        x3 = self.conv_bloc(x, filters, x3_kernel, [1,2,4], initializer)
        x4 = self.conv_bloc(x, filters, x4_kernel, [1,4,8], initializer)

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

        self.architecture.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy', F1Score(average='micro', num_classes=self.number_of_models_involved)])

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

    def plot_confusion_matrix(self, normalized=True):
        trajectories = self.simulator().simulate_trajectories_by_category(self.hyperparameters['validation_set_size'], self.trajectory_length, self.models_involved_in_predictive_model, self.trajectory_time)

        ground_truth = np.argmax(self.transform_trajectories_to_output(trajectories), axis=-1)
        Y_predicted = self.predict(trajectories)

        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=Y_predicted)

        if normalized:
            confusion_mat = confusion_mat.astype(
                'float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        labels = [a_tuple[0] for a_tuple in self.models_involved_in_predictive_model]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        # Plot matrix
        plt.title(f'Confusion Matrix (F1={round(f1_score(ground_truth, Y_predicted, average="micro"),2)})')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        #plt.show()
        plt.savefig(str(self)+'.jpg')
        plt.clf()

    def __str__(self):
        return f"wavenet_tcn_theoretical_model_classifier_{self.trajectory_length}_simulation_{self.simulator().STRING_LABEL}"
