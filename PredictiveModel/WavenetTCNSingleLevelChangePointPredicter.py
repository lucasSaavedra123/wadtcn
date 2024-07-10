from keras.layers import Dense, Input, Average, Conv1D, TimeDistributed
from keras.models import Model
from keras.losses import BinaryCrossentropy, BinaryCrossentropy, BinaryFocalCrossentropy 
from keras.metrics.accuracy_metrics import CategoricalAccuracy
from keras.metrics.confusion_metrics import AUC, Recall, Precision
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import glob as glob
from andi_datasets.datasets_challenge import _defaults_andi2
import os


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

class WavenetTCNSingleLevelChangePointPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def default_hyperparameters(self, **kwargs):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':999}

    @classmethod
    def selected_hyperparameters(self):
        return {'lr': 0.0001, 'batch_size': 32, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':999}

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
        return ['No Change', 'Change']

    def transform_trajectories_to_output(self, trajectories):
        d = transform_trajectories_to_single_level_diffusion_coefficient(self, trajectories)
        m = transform_trajectories_to_single_level_model_as_number(self, trajectories)
        h = transform_trajectories_to_single_level_hurst_exponent(self, trajectories)
        output = np.zeros(d.shape)
        output[:,1:] = np.diff(d) + np.diff(m) + np.diff(h)
        output = (output != 0).astype(float)
        return output

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories)
        return X

    def predict(self, trajectories):
        return (self.architecture.predict(self.transform_trajectories_to_input(trajectories)) > 0.5).astype(int)

    @property
    def type_name(self):
        return 'wavenet_changepoint_detector'

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}"

    def build_network(self):
        number_of_features = 2
        wavenet_filters = 32
        dff = 320
        number_of_passes = 2

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

        x_1 = x
        #Following code is similar to Requena, 2023.
        for _ in range(number_of_passes):
            x = EncoderLayer(d_model=wavenet_filters*5, num_heads=4, dff=dff, dropout_rate=0.1)(x)
        x = Add()([x_1, x])
        
        x = LayerNormalization()(x)
        x_1 = x
        x = FeedForward(wavenet_filters*5, dff, 0.1)(x)
        x = Add()([x_1, x])
        x = LayerNormalization()(x)
        
        x = FeedForward(wavenet_filters*5, dff, 0.1)(x)

        #x = Conv1D(filters=wavenet_filters*5, kernel_size=3, padding='causal', activation='relu', kernel_initializer=initializer)(x)
        output = Dense(units=len(self.models_involved_in_predictive_model), activation='sigmoid', name='change_point_detection')(x)

        self.architecture = Model(inputs=inputs, outputs=output)

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

        self.architecture.compile(optimizer= optimizer, loss=BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=True), metrics=[CategoricalAccuracy(), AUC(), Recall(), Precision()])#, metrics=[weighted_binary_crossentropy])

    def prepare_dataset(self, set_size, file_label='', get_from_cache=False):
        trajectories = self.simulator().simulate_phenomenological_trajectories_for_classification_training(set_size, self.trajectory_length, self.trajectory_time, get_from_cache=get_from_cache, file_label=file_label, enable_parallelism=True, type_of_simulation='models_phenom')
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

        for X_val_i in range(X_val.shape[0]):
            X_val[X_val_i] += np.random.randn(*X_val[X_val_i].shape) * np.random.uniform(0.5,1.5) * _defaults_andi2().sigma_noise

        number_of_training_trajectories = len(glob.glob('./2ndAndiTrajectories/*_X_cp.npy'))

        def custom_prepare_dataset(batch_size):            
            X, Y = [], []
            while len(X) < batch_size:
                selected_model = np.random.randint(len(self.models_involved_in_predictive_model))
                retry = True
                while retry:
                    trajectory_id = np.random.randint(number_of_training_trajectories)
                    Y_i = np.load(os.path.join('./2ndAndiTrajectories', f'{trajectory_id}_Y_cp.npy'))
                    simple_Y_i = np.unique(np.argmax(Y_i,axis=2))
                    if selected_model == 2:
                        retry = not (len(simple_Y_i) == 1 and 2 in simple_Y_i)
                    else:
                        retry = not (len(simple_Y_i) == 2 and selected_model in simple_Y_i)

                X.append(np.load(os.path.join('./2ndAndiTrajectories', f'{trajectory_id}_X_cp.npy')))
                Y.append(Y_i)
                X[-1] += np.random.randn(*X[-1].shape) * np.random.uniform(0.5,1.5) * _defaults_andi2().sigma_noise

            X = np.concatenate(X)
            Y = np.concatenate(Y)
            return X, Y

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

    def plot_confusion_matrix(self, trajectories=None, normalized=True, sigma=0):
        if trajectories is None:
            trajectories = self.simulator().simulate_phenomenological_trajectories_for_classification_training(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, get_from_cache=True, file_label='val', type_of_simulation='models_phenom')

        for t in trajectories:
            t.x = (np.array(t.x) + np.random.randn(t.length)*sigma).tolist()
            t.y = (np.array(t.y) + np.random.randn(t.length)*sigma).tolist()

        result = self.predict(trajectories)
        result = np.argmax(result,axis=2)

        ground_truth = []
        predicted = []

        for i, ti in enumerate(trajectories):
            ground_truth += np.argmax(self.transform_trajectories_to_output([ti]), axis=2)[0].tolist()
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

    def plot_single_level_prediction(self, limit=10, sigma=0):
        trajectories = self.simulator().simulate_phenomenological_trajectories_for_classification_training(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, get_from_cache=True, file_label='val')

        for t in trajectories:
            t.x = (np.array(t.x) + np.random.randn(t.length)*sigma).tolist()
            t.y = (np.array(t.y) + np.random.randn(t.length)*sigma).tolist()

        np.random.shuffle(trajectories)
        result = self.predict(trajectories[:limit])
        idxs = np.arange(0,limit, 1)

        for i in idxs:
            ti = trajectories[i]
            fig, ax = plt.subplots(1,2)
            ax[0].plot(ti.get_noisy_x(), ti.get_noisy_y())
            ax[1].set_title('State')
            ax[1].plot(ti.info['state_t'], color='black')
            ax[1].plot(np.argmax(result[i], axis=1), color='red')
            ax[1].set_ylim([-1,4])
            #ax.set_yticklabels([s.capitalize() for s in self.models_involved_in_predictive_model])
            plt.show()
