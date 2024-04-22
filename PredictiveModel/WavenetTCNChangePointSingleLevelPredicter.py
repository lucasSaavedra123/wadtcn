from keras.layers import Dense, Input, Average, Conv1D, TimeDistributed
from keras.models import Model
from keras.losses import BinaryCrossentropy, BinaryCrossentropy, BinaryFocalCrossentropy 
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd

from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
from tensorflow import device, config


#https://medium.com/the-owl/weighted-binary-cross-entropy-losses-in-keras-e3553e28b8db
def weighted_binary_crossentropy(target, output, weights=[1,10]):
    target = tf.convert_to_tensor(tf.reshape(target, [-1]))
    output = tf.convert_to_tensor(tf.reshape(output, [-1]))
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return tf.reduce_mean(-bce,axis=-1)

class WavenetTCNChangePointSingleLevelPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        hyperparameters = {'lr': 0.001, 'batch_size': 512, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 100}
        return hyperparameters

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [8, 16, 32, 64],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return ['No Change', 'Change']

    def predict(self, trajectories):
        return (self.architecture.predict(self.transform_trajectories_to_input(trajectories)) > 0.5).astype(int)

    def transform_trajectories_to_output(self, trajectories):
        offset = 2
        d = transform_trajectories_to_single_level_diffusion_coefficient(self, trajectories)
        m = transform_trajectories_to_single_level_model_as_number(self, trajectories)
        h = transform_trajectories_to_single_level_hurst_exponent(self, trajectories)
        output = np.zeros(d.shape)
        output[:,1:] = np.diff(d) + np.diff(m) + np.diff(h)
        output = (output != 0).astype(float)

        #for i in range(len(trajectories)):
        #    ones_index = np.where(output[i] == 1)[0]  # Obtenemos los índices donde hay '1'
        #    for index in ones_index:
        #        output[max(0, index - offset):min(len(output[i]), index + offset + 1)] = 1

        return output

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories)

        if self.wadnet_tcn_encoder is not None:
            X = self.wadnet_tcn_encoder.predict(X, verbose=0)
        return X

    def build_network(self):
        if self.wadnet_tcn_encoder is None:
            number_of_features = 2
            inputs = Input(shape=(self.trajectory_length, number_of_features))
            wavenet_filters = 16
            dilation_depth = 1
            initializer = 'he_normal'

            x = WaveNetEncoder(wavenet_filters, dilation_depth, initializer=initializer)(inputs)
            unet_1 = Unet((self.trajectory_length, wavenet_filters), '1d', 2, unet_index=1)(x)
            unet_2 = Unet((self.trajectory_length, wavenet_filters), '1d', 3, unet_index=2)(x)
            unet_3 = Unet((self.trajectory_length, wavenet_filters), '1d', 4, unet_index=3)(x)
            unet_4 = Unet((self.trajectory_length, wavenet_filters), '1d', 9, unet_index=4)(x)

            x = concatenate([unet_1, unet_2, unet_3, unet_4])
            #output_network = Conv1D(1, 3, 1, padding='same', activation='sigmoid')(x)
            output_network = TimeDistributed(Dense(units=1, activation='sigmoid'))(x)
            self.architecture = Model(inputs=inputs, outputs=output_network)

        else:
            inputs = Input(shape=(64))
            x = Dense(units=128, activation='selu')(inputs)
            output_network = Dense(units=1, activation='sigmoid')(x)
            self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
            learning_rate=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )
        """
        from tensorflow import reduce_mean, square, reshape, abs
        def custom_mse(y_true, y_pred):
            y_true = reshape(y_true, [-1])
            y_pred = reshape(y_pred, [-1])
            return reduce_mean(square(y_true - y_pred))

        def custom_mae(y_true, y_pred):
            y_true = reshape(y_true, [-1])
            y_pred = reshape(y_pred, [-1])
            return reduce_mean(abs(y_true - y_pred))

        self.architecture.compile(optimizer=optimizer, loss=custom_mse, metrics=[custom_mse, custom_mae])
        """

        self.architecture.compile(optimizer= optimizer, loss=BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=True))#, metrics=[weighted_binary_crossentropy])

    @property
    def type_name(self):
        return 'wavenet_single_level_hurst_exponent'

    def plot_bias(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten() * 2
        predicted = self.predict(trajectories).flatten() * 2

        plot_bias(ground_truth, predicted, symbol='alpha')

    def plot_predicted_and_ground_truth_distribution(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten() * 2
        predicted = self.predict(trajectories).flatten() * 2

        plot_predicted_and_ground_truth_distribution(ground_truth, predicted)

    def plot_predicted_and_ground_truth_histogram(self):
        trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories).flatten() * 2
        predicted = self.predict(trajectories).flatten() * 2

        plot_predicted_and_ground_truth_histogram(ground_truth, predicted, range=[[0,2],[0,2]])

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}"

    def change_network_length_to(self, new_length):
        self.trajectory_length = new_length
        old_architecture = self.architecture
        self.build_network()

        for i, _ in enumerate(old_architecture.layers):
            if i == 0:
                continue
            else:
                if 'unet' in old_architecture.layers[i].name:
                    for j, _ in enumerate(old_architecture.layers[i].layers):
                        if j == 0 or old_architecture.layers[i].layers[j].name == 'LeakyReLU':
                            continue
                        else:
                            self.architecture.layers[i].layers[j].set_weights(old_architecture.layers[i].layers[j].get_weights())
                else:
                    self.architecture.layers[i].set_weights(old_architecture.layers[i].get_weights())

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

        X_train, Y_train = self.prepare_dataset(TRAINING_SET_SIZE_PER_EPOCH, file_label='train', get_from_cache=True)
        X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH, file_label='val', get_from_cache=True)

        with device(device_name):
            history_training_info = self.architecture.fit(
                X_train, Y_train,
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

    def prepare_dataset(self, set_size, file_label='', get_from_cache=False):
        trajectories = self.simulator().simulate_phenomenological_trajectories(set_size, self.trajectory_length, self.trajectory_time, get_from_cache=get_from_cache, file_label=file_label)
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def plot_confusion_matrix(self, trajectories=None, normalized=True):
        if trajectories is None:
            trajectories = self.simulator().simulate_phenomenological_trajectories(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, get_from_cache=True, file_label='val')

        result = self.predict(trajectories)
        #result = np.argmax(result,axis=2)

        ground_truth = []
        predicted = []

        for i, ti in enumerate(trajectories):
            ground_truth += self.transform_trajectories_to_output([ti])[0].tolist()
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
