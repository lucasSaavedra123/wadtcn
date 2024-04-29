import numpy as np
from keras.layers import Dense, Input, TimeDistributed
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.metrics import confusion_matrix, f1_score

from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
import pandas as pd
from keras.callbacks import EarlyStopping
from tensorflow import device, config
import keras.backend as K

class WavenetTCNModelSingleLevelPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        return {'lr': 0.001, 'batch_size': 512, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 1}

    @classmethod
    def selected_hyperparameters(self):
        return {'lr': 0.001, 'batch_size': 512, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 1}

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [128, 256, 512, 1024],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return ['trap', 'confined', 'free', 'directed']

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories))

    def transform_trajectories_to_output(self, trajectories):
        Y1 = transform_trajectories_to_single_level_model(self, trajectories)
        Y2 = transform_trajectories_to_single_level_hurst_exponent(self, trajectories)
        Y3 = transform_trajectories_to_single_level_diffusion_coefficient(self, trajectories)
        Y3[Y3==0] = 1e-12
        return Y1, Y2, np.log10(Y3)

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)
        return X

    def build_network(self):
        number_of_features = 2
        inputs = Input(shape=(self.trajectory_length, number_of_features))
        wavenet_filters = 16
        dilation_depth = 8
        initializer = 'he_normal'

        x = WaveNetEncoder(wavenet_filters, dilation_depth, initializer=initializer)(inputs)
        unet_1 = Unet((self.trajectory_length, wavenet_filters), '1d', 2, unet_index=1, skip_last_block=True)(x)
        unet_2 = Unet((self.trajectory_length, wavenet_filters), '1d', 3, unet_index=2, skip_last_block=True)(x)
        unet_3 = Unet((self.trajectory_length, wavenet_filters), '1d', 4, unet_index=3, skip_last_block=True)(x)
        unet_4 = Unet((self.trajectory_length, wavenet_filters), '1d', 9, unet_index=4, skip_last_block=True)(x)            
        unet_5 = Unet((self.trajectory_length, wavenet_filters), '1d', 13, unet_index=5, skip_last_block=True)(x)            
        unet_6 = Unet((self.trajectory_length, wavenet_filters), '1d', 17, unet_index=6, skip_last_block=True)(x)            

        x = concatenate([unet_1, unet_2, unet_3, unet_4, unet_5, unet_6])

        model_classification = Conv1D(16, 3, 1, padding='causal')(x)
        model_classification = LeakyReLU()(model_classification)
        model_classification = TimeDistributed(Dense(units=4, activation='softmax'), name='model_classification_output')(model_classification)

        def custom_tanh_1(x):
            return (K.tanh(x)+1)/2

        alpha_regression = Conv1D(16, 3, 1, padding='causal')(x)
        alpha_regression = LeakyReLU()(alpha_regression)
        alpha_regression = TimeDistributed(Dense(units=1, activation=custom_tanh_1), name='alpha_regression_output')(alpha_regression)

        def custom_tanh_2(x):
            return ((K.tanh(x)+1)*9)-12

        d_regression = Conv1D(16, 3, 1, padding='causal')(x)
        d_regression = LeakyReLU()(d_regression)
        d_regression = TimeDistributed(Dense(units=1, activation=custom_tanh_2), name='d_regression_output')(d_regression)

        self.architecture = Model(inputs=inputs, outputs=[model_classification, alpha_regression, d_regression])

        optimizer = Adam(
            learning_rate=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )

        loss_parameter = {
            'model_classification_output': 'categorical_crossentropy',
            'alpha_regression_output': 'mse',
            'd_regression_output': 'mse'
        }

        metrics_parameter = {
            'model_classification_output': 'categorical_accuracy',
            'alpha_regression_output': 'mae',
            'd_regression_output': 'mae'
        }

        self.architecture.compile(optimizer=optimizer, loss=loss_parameter, metrics=metrics_parameter)

    @property
    def type_name(self):
        return 'wavenet_single_level_model'

    def prepare_dataset(self, set_size, file_label='', get_from_cache=False):
        trajectories = self.simulator().simulate_phenomenological_trajectories(set_size, self.trajectory_length, self.trajectory_time, get_from_cache=get_from_cache, file_label=file_label)
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

        #X_train, Y_train = self.prepare_dataset(TRAINING_SET_SIZE_PER_EPOCH, file_label='train', get_from_cache=True)
        #X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH, file_label='val', get_from_cache=True)

        #Y1_train = Y_train[0]
        #Y2_train = Y_train[1]
        #Y3_train = Y_train[2]
        #Y1_val = Y_val[0]
        #Y2_val = Y_val[1]
        #Y3_val = Y_val[2]

        #np.save('xv.npy', X_val)
        #np.save('xt.npy', X_train)
        #np.save('yv0.npy', Y_val[0])
        #np.save('yt0.npy', Y_train[0])
        #np.save('yv1.npy', Y_val[1])
        #np.save('yt1.npy', Y_train[1])
        #np.save('yv2.npy', Y_val[2])
        #np.save('yt2.npy', Y_train[2])
        #exit()
        X_train, Y1_train, Y2_train, Y3_train = np.load('xt.npy'), np.load('yt0.npy'), np.load('yt1.npy'), np.load('yt2.npy')
        X_val, Y1_val, Y2_val, Y3_val = np.load('xv.npy'), np.load('yv0.npy'), np.load('yv1.npy'), np.load('yv2.npy')

        with device(device_name):
            history_training_info = self.architecture.fit(
                X_train, [Y1_train, Y2_train, Y3_train],
                epochs=real_epochs,
                callbacks=callbacks,
                batch_size=self.hyperparameters['batch_size'],
                validation_data=[X_val, [Y1_val, Y2_val, Y3_val]],
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
        result = self.predict(trajectories)
        result = result[2]#np.argmax(result,axis=2)
        idxs = np.arange(0,len(trajectories), 1)
        np.random.shuffle(idxs)

        for i in idxs[:limit]:
            ti = trajectories[i]
            plt.plot(np.array(ti.info['d_t']), color='black')
            plt.plot(result[i, :], color='red')
            #plt.ylim([-1,1])
            plt.show()
