from keras.layers import Dense, Input, Average, Conv1D, TimeDistributed
from keras.models import Model
from keras.losses import BinaryCrossentropy, BinaryCrossentropy, BinaryFocalCrossentropy 
#from keras.metrics.accuracy_metrics import CategoricalAccuracy
#from keras.metrics.confusion_metrics import AUC, Recall, Precision
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import glob as glob
from andi_datasets.datasets_challenge import _defaults_andi2
import os
import random


from .PredictiveModel import PredictiveModel
from .model_utils import *
from CONSTANTS import *
from Trajectory import Trajectory
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
        return {'lr': 0.0001, 'batch_size': 128, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':999}

    @classmethod
    def selected_hyperparameters(self):
        return {'lr': 0.0001, 'batch_size': 128, 'amsgrad': False, 'epsilon': 1e-06, 'epochs':999}

    @classmethod
    def default_hyperparameters_analysis(self):
        return {
            'lr': [1e-2, 1e-3, 1e-4, 1e-5],
            'amsgrad': [False, True],
            'batch_size': [32, 64, 128, 256],
            'epsilon': [1e-6, 1e-7, 1e-8]
        }

    @property
    def models_involved_in_predictive_model(self):
        return ['No Change', 'Change']

    def transform_trajectories_to_output(self, trajectories):
        if self.simulator.STRING_LABEL == 'andi2':
            d = transform_trajectories_to_single_level_diffusion_coefficient(self, trajectories)
            m = transform_trajectories_to_single_level_model_as_number(self, trajectories)
            h = transform_trajectories_to_single_level_hurst_exponent(self, trajectories)
            output = np.zeros(d.shape)
            output[:,1:] = np.diff(d) + np.diff(m) + np.diff(h)
            output = (output != 0).astype(float)
            return output
        elif self.simulator.STRING_LABEL == 'andi':
            output = np.zeros((len(trajectories), trajectories[0].length))
            for ti, t in enumerate(trajectories):
                output[ti,:] = [0] * int(t.info['change_point_time']-1) + [1] + [0] * int(t.length - t.info['change_point_time'])
                #output[ti,:] = [0] * int(t.info['change_point_time']) + [1] * int(t.length - t.info['change_point_time'])

            return output

    def transform_trajectories_to_input(self, trajectories):
        X = transform_trajectories_into_raw_trajectories(self, trajectories, normalize=True)
        return X

    def predict(self, trajectories, apply_threshold=True):
        predictions = self.architecture.predict(self.transform_trajectories_to_input(trajectories))

        decision_threshold = 0.48332304 if self.simulator.STRING_LABEL == 'andi' else 0.031339474

        if apply_threshold:
            predictions = (predictions > decision_threshold).astype(int)
        return predictions

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

        x = Transformer(2,4,wavenet_filters*5,320)(x)

        #x = Conv1D(filters=wavenet_filters*5, kernel_size=3, padding='causal', activation='relu', kernel_initializer=initializer)(x)
        output = Dense(units=1, activation='sigmoid', name='change_point_detection')(x)

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

        def loss(t,o):
            #return weighted_binary_crossentropy(t,o,weights=[1/(200*2), 199/(200*2)])
            return weighted_binary_crossentropy(t,o,weights=[1,199])
        self.architecture.compile(optimizer= optimizer, loss=loss, metrics=['auc'])
        #self.architecture.compile(optimizer= optimizer, loss=BinaryCrossentropy(from_logits=False), metrics=[CategoricalAccuracy(), AUC(), Recall(), Precision()])
        #self.architecture.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=[CategoricalAccuracy(), AUC(), Recall(), Precision()])

    @property
    def dataset_type(self):
        return 'classifier'

    def prepare_dataset(self, set_size, files=None):
        if self.simulator.STRING_LABEL == 'andi2':
            trajectories = np.random.choice(files,size=set_size, replace=False).tolist()
            for i in range(len(trajectories)):
                df = pd.read_csv(trajectories[i])
                df = df.sort_values('t', ascending=True)

                sigma = np.random.uniform(0,2)

                trajectories[i] = Trajectory(
                    x=df['x'].tolist(),
                    y=df['y'].tolist(),
                    noise_x=np.random.randn(len(df)) * sigma,
                    noise_y=np.random.randn(len(df)) * sigma,
                    info={
                        'd_t':df['d_t'].tolist(),
                        'alpha_t':df['alpha_t'].tolist(),
                        'state_t':df['state_t'].tolist()
                    }
                )
        elif self.simulator.STRING_LABEL == 'andi':
            trajectories = self.simulator().simulate_segmentated_trajectories(set_size, self.trajectory_length, self.trajectory_time)

        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def plot_confusion_matrix(self, trajectories=None, normalized=True, sigma=0):
        if trajectories is None:
            if self.simulator.STRING_LABEL == 'andi2':
                trajectories = self.simulator().simulate_phenomenological_trajectories_for_classification_training(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, get_from_cache=True, file_label='val', type_of_simulation='models_phenom')
            elif self.simulator.STRING_LABEL == 'andi':
                trajectories = self.simulator().simulate_segmentated_trajectories(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time)

        for t in trajectories:
            t.x = (np.array(t.x) + np.random.randn(t.length)*sigma).tolist()
            t.y = (np.array(t.y) + np.random.randn(t.length)*sigma).tolist()

        result = self.predict(trajectories)

        ground_truth = []
        predicted = []

        for i, ti in enumerate(trajectories):
            ground_truth += self.transform_trajectories_to_output([ti])[0].tolist()
            predicted += result[i,:,0].tolist()

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
