import os
import numpy as np
from keras.layers import Dense, Input, TimeDistributed
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import glob
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.metrics import confusion_matrix, f1_score
from Trajectory import Trajectory
from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
import pandas as pd
from keras.callbacks import EarlyStopping
from tensorflow import device, config
import keras.backend as K

class WavenetTCNMultiTaskClassifierSingleLevelPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':100}

    @classmethod
    def selected_hyperparameters(self):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':100}

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [32, 64, 128, 256, 512, 1024],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return ['trap', 'confined', 'free', 'directed']

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories), verbose=0)

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_single_level_model(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories)
        #X_displacements = transform_trajectories_into_displacements(self, trajectories)
        #X_turning_angle = transform_trajectories_into_turning_angle(self, trajectories)

        #X = np.zeros((len(trajectories), self.trajectory_length, 3))
        #X[:,1:,0:2] = X_displacements
        #X[:,2:,2:] = X_turning_angle
        #if self.wadnet_tcn_encoder is not None:
        #    X = self.wadnet_tcn_encoder.predict(X, verbose=0)
        return X

    def build_network(self):
        number_of_features = 2
        wavenet_filters = 32
        dilation_depth = 8
        initializer = 'he_normal'
        x1_kernel = 4
        x2_kernel = 2
        x3_kernel = 3
        x4_kernel = 10
        x5_kernel = 20

        dilation_depth = 8

        inputs = Input(shape=(None, number_of_features))

        x = WaveNetEncoder(wavenet_filters, dilation_depth, initializer=initializer)(inputs)

        x1 = convolutional_block(self, x, wavenet_filters, x1_kernel, [1,2,4], initializer)
        x2 = convolutional_block(self, x, wavenet_filters, x2_kernel, [1,2,4], initializer)
        x3 = convolutional_block(self, x, wavenet_filters, x3_kernel, [1,2,4], initializer)
        x4 = convolutional_block(self, x, wavenet_filters, x4_kernel, [1,4,8], initializer)

        x5 = Conv1D(filters=wavenet_filters, kernel_size=x5_kernel, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x5 = BatchNormalization()(x5)

        x = concatenate(inputs=[x1, x2, x3, x4, x5])

        x = Conv1D(filters=wavenet_filters*5, kernel_size=3, padding='causal', activation='relu', kernel_initializer=initializer)(x)
        output = TimeDistributed(Dense(units=len(self.models_involved_in_predictive_model), activation='softmax'), name='model_classification_output')(x)

        self.architecture = Model(inputs=inputs, outputs=output)

        optimizer = Adam(
            learning_rate=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )

        self.architecture.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='categorical_accuracy')

    @property
    def type_name(self):
        return 'wavenet_single_level_classifier_model'

    def prepare_dataset(self, set_size, file_label='', get_from_cache=False):
        trajectories = self.simulator().simulate_phenomenological_trajectories(set_size, self.trajectory_length, self.trajectory_time, get_from_cache=get_from_cache, file_label=file_label, enable_parallelism=True)
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def fit(self):

        if not self.trained:
            self.build_network()
            real_epochs = self.hyperparameters['epochs']
        else:
            real_epochs = self.hyperparameters['epochs'] - len(self.history_training_info['loss'])

        self.architecture.summary()

        if self.early_stopping:
            callbacks = [EarlyStopping(
                monitor="val_loss",
                min_delta=1e-3,
                patience=5,
                verbose=1,
                mode="min")]
        else:
            callbacks = []

        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH, file_label='val', get_from_cache=True)

        number_of_training_trajectories = len(glob.glob('./2ndAndiTrajectories/*.npy'))/4

        def custom_prepare_dataset(batch_size):            
            trajectories_ids = np.random.randint(number_of_training_trajectories, size=batch_size)

            X, Y0 = [], []

            for trajectory_id in trajectories_ids:
                X.append(np.load(os.path.join('./2ndAndiTrajectories', f'{trajectory_id}_X.npy')))
                Y0.append(np.load(os.path.join('./2ndAndiTrajectories', f'{trajectory_id}_Y0.npy')))

            X = np.concatenate(X)
            Y0 = np.concatenate(Y0)

            return X, Y0

        with device(device_name):
            history_training_info = self.architecture.fit(
                TrackGenerator(TRAINING_SET_SIZE_PER_EPOCH//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], custom_prepare_dataset),
                epochs=real_epochs,
                callbacks=callbacks,
                batch_size=self.hyperparameters['batch_size'],
                validation_data=[X_val, Y_val],
                shuffle=True
            ).history

        if self.trained:
            for dict_key in history_training_info:
                self.history_training_info[dict_key] += history_training_info[dict_key]
        else:
            self.history_training_info = history_training_info
            self.trained = True

    def plot_confusion_matrix(self, trajectories=None, normalized=True):
        if trajectories is None:
            trajectories = self.simulator().simulate_phenomenological_trajectories(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, get_from_cache=True, file_label='val')

        result = self.predict(trajectories)
        result = np.argmax(result[0],axis=2)

        ground_truth = []
        predicted = []

        for i, ti in enumerate(trajectories):
            ground_truth += np.argmax(self.transform_trajectories_to_output([ti])[0], axis=2)[0].tolist()
            predicted += result[i,:].tolist()

        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predicted)

        confusion_mat = np.round(confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis], 2) if normalized else confusion_mat

        labels = [state.upper() for state in self.models_involved_in_predictive_model]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        plt.title(f'Confusion Matrix (F1={round(f1_score(ground_truth, predicted, average="micro"),2)})')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        plt.show()

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}"

    def plot_single_level_prediction(self, limit=10):
        trajectories = self.simulator().simulate_phenomenological_trajectories(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, get_from_cache=True, file_label='val')
        np.random.shuffle(trajectories)
        result = self.predict(trajectories[:limit])
        idxs = np.arange(0,limit, 1)

        fig, ax = plt.subplots()

        for i in idxs:
            ti = trajectories[i]
            ax.set_title('State')
            ax.plot(ti.info['state_t'], color='black')
            ax.plot(np.argmax(result[0][i, :], axis=1), color='red')
            ax.set_ylim([-1,5])
            ax.set_xticklabels([s.capitalize() for s in self.models_involved_in_predictive_model])
            plt.show()
