import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras import Sequential
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate, Add, Multiply, Layer, GlobalAveragePooling1D
from keras.models import Model

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def transform_trajectories_into_displacements(predictive_model, trajectories, normalize=False):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length-1, 2))

    def axis_adaptation_to_net(axis_data, track_length):
        axis_reshaped = np.reshape(axis_data, newshape=[1, len(axis_data)])
        axis_reshaped = axis_reshaped - np.mean(axis_reshaped)
        axis_diff = np.diff(axis_reshaped[0, :track_length])
        return axis_diff

    for index, trajectory in enumerate(trajectories):
        X[index, :, 0] = axis_adaptation_to_net(trajectory.get_noisy_x(), predictive_model.trajectory_length)
        X[index, :, 1] = axis_adaptation_to_net(trajectory.get_noisy_y(), predictive_model.trajectory_length)

        if predictive_model.simulator.STRING_LABEL == 'andi' or normalize:
            X[index, :, 0] = (X[index, :, 0] - np.mean(X[index, :, 0]))/(np.std(X[index, :, 0]) if np.std(X[index, :, 0])!= 0 else 1)
            X[index, :, 1] = (X[index, :, 1] - np.mean(X[index, :, 1]))/(np.std(X[index, :, 1]) if np.std(X[index, :, 1])!= 0 else 1)

    return X

def transform_trajectories_into_displacements_with_time(predictive_model, trajectories, normalize=False):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length-1, 3))

    def axis_adaptation_to_net(axis_data, track_length):
        axis_reshaped = np.reshape(axis_data, newshape=[1, len(axis_data)])
        axis_reshaped = axis_reshaped - np.mean(axis_reshaped)
        axis_diff = np.diff(axis_reshaped[0, :track_length])
        return axis_diff

    for index, trajectory in enumerate(trajectories):
        X[index, :, 0] = axis_adaptation_to_net(trajectory.get_noisy_x(), predictive_model.trajectory_length)
        X[index, :, 1] = axis_adaptation_to_net(trajectory.get_noisy_y(), predictive_model.trajectory_length)

        if predictive_model.simulator.STRING_LABEL == 'andi' or normalize:
            X[index, :, 0] = (X[index, :, 0] - np.mean(X[index, :, 0]))/(np.std(X[index, :, 0]) if np.std(X[index, :, 0])!= 0 else 1)
            X[index, :, 1] = (X[index, :, 1] - np.mean(X[index, :, 1]))/(np.std(X[index, :, 1]) if np.std(X[index, :, 1])!= 0 else 1)

        X[index, :, 2] = np.diff(trajectory.get_time())

    return X

def transform_trajectories_to_mean_square_displacement_segments(predictive_model, trajectories):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length-2, 1))

    for index, trajectory in enumerate(trajectories):
        _, msd, _, _, _ = trajectory.temporal_average_mean_squared_displacement()
        X[index, :, 0] = msd

    return X

def transform_trajectories_into_squared_differences(predictive_model, trajectories, normalize=False):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length-1, 1))

    for index, trajectory in enumerate(trajectories):
        r = np.sqrt((trajectory.get_noisy_x()**2) + (trajectory.get_noisy_y()**2))
        diff = np.diff(r)
        diff_sq = diff**2

        if predictive_model.simulator.STRING_LABEL == 'andi' or normalize:
            diff_sq = diff_sq - np.mean(diff_sq)

        X[index, :, 0] = diff_sq

    return X

def transform_trajectories_into_raw_trajectories(predictive_model, trajectories, normalize=False):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length, 2))

    for index, trajectory in enumerate(trajectories):
        X[index, :, 0] = trajectory.get_noisy_x() - np.mean(trajectory.get_noisy_x())
        X[index, :, 1] = trajectory.get_noisy_y() - np.mean(trajectory.get_noisy_y())

        if predictive_model.simulator.STRING_LABEL == 'andi' or normalize:
            X[index, :, 0] = X[index, :, 0]/(np.std(X[index, :, 0]) if np.std(X[index, :, 0])!= 0 else 1)
            X[index, :, 1] = X[index, :, 1]/(np.std(X[index, :, 1]) if np.std(X[index, :, 1])!= 0 else 1)

    return X

def transform_trajectories_into_states(predictive_model, trajectories):
    Y = np.empty((len(trajectories), predictive_model.trajectory_length))

    for index, trajectory in enumerate(trajectories):
        if 'state' in trajectory.info:
            Y[index, :] = trajectory.info['state']
        else:
            Y[index, :] = np.zeros((predictive_model.trajectory_length))

    return Y

def transform_trajectories_to_categorical_vector(predictive_model, trajectories):
    Y_as_vectors = np.empty((len(trajectories), predictive_model.number_of_models_involved))

    for index, trajectory in enumerate(trajectories):
        Y_as_vectors[index, :] = to_categorical(predictive_model.model_to_label(trajectory.model_category), num_classes=predictive_model.number_of_models_involved)

    return Y_as_vectors


def transform_trajectories_to_change_point_time(predictive_model, trajectories):
    Y_as_vectors = np.empty((len(trajectories), 2))

    """
    np.sin((2*np.pi*Y[:,1])/traj_length) # sine of the switching time
        label_inf[:,3]=np.cos((2*np.pi*Y[:,1])/traj_length)
    """

    for index, trajectory in enumerate(trajectories):
        Y_as_vectors[index, 0] = np.sin((2*np.pi*trajectory.info['change_point_time'])/predictive_model.trajectory_length)
        Y_as_vectors[index, 1] = np.cos((2*np.pi*trajectory.info['change_point_time'])/predictive_model.trajectory_length)

    return Y_as_vectors

def transform_trajectories_to_anomalous_exponent(predictive_model, trajectories):
    Y = np.empty((len(trajectories), 1))

    for index, trajectory in enumerate(trajectories):
        Y[index, 0] = trajectory.anomalous_exponent 

    return Y

def transform_trajectories_to_hurst_exponent(predictive_model, trajectories):
    Y = np.empty((len(trajectories), 1))

    for index, trajectory in enumerate(trajectories):
        Y[index, 0] = trajectory.hurst_exponent()

    return Y

def transform_trajectories_to_diffusion_coefficient(predictive_model, trajectories):
    Y = np.empty((len(trajectories), 1))

    for index, trajectory in enumerate(trajectories):
        Y[index, 0] = trajectory.info['diffusion_coefficient']

    return Y

def convolutional_block(predictive_model, original_x, filters, kernel_size, dilation_rates, initializer):
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

class WaveNetEncoder(Layer):
    def __init__(self, filters, dilation_depth, initializer='he_normal'):
        super().__init__()
        self.dilation_depth = dilation_depth
        self.filters = filters

        wavenet_dilations = [2**i for i in range(self.dilation_depth)]
        self.conv_1d_tanh_layers = [Conv1D(self.filters, kernel_size=3, dilation_rate=dilation, padding='causal', activation='tanh', kernel_initializer=initializer) for dilation in wavenet_dilations]
        self.conv_1d_sigm_layers = [Conv1D(self.filters, kernel_size=3, dilation_rate=dilation, padding='causal', activation='sigmoid', kernel_initializer=initializer) for dilation in wavenet_dilations]

        self.first_conv_layer = Conv1D(self.filters, 3, padding='causal', kernel_initializer=initializer)
        self.wavenet_enconder_convs = [Conv1D(self.filters, kernel_size=1, padding='causal', kernel_initializer=initializer) for _ in wavenet_dilations]

        self.last_batch_normalization = BatchNormalization()

    def call(self, inputs):
        x = self.first_conv_layer(inputs)

        layers_to_add = [x]

        for i in range(self.dilation_depth):
            tanh_out = self.conv_1d_tanh_layers[i](x)
            sigm_out = self.conv_1d_sigm_layers[i](x)

            x = Multiply()([tanh_out, sigm_out])
            x = self.wavenet_enconder_convs[i](x)

            layers_to_add.append(x)

        x = Add()(layers_to_add)
        x = self.last_batch_normalization(x)

        return x

def build_wavenet_tcn_classifier_for(predictive_model, filters=64, number_of_features=2):
    initializer = 'he_normal'
    x1_kernel = 4
    x2_kernel = 2
    x3_kernel = 3
    x4_kernel = 10
    x5_kernel = 20

    dilation_depth = 8

    inputs = Input(shape=(predictive_model.trajectory_length-1, number_of_features))

    x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

    x1 = convolutional_block(predictive_model, x, filters, x1_kernel, [1,2,4], initializer)
    x2 = convolutional_block(predictive_model, x, filters, x2_kernel, [1,2,4], initializer)
    x3 = convolutional_block(predictive_model, x, filters, x3_kernel, [1,2,4], initializer)
    x4 = convolutional_block(predictive_model, x, filters, x4_kernel, [1,4,8], initializer)

    x5 = Conv1D(filters=filters, kernel_size=x5_kernel, padding='same', activation='relu', kernel_initializer=initializer)(x)
    x5 = BatchNormalization()(x5)

    x = concatenate(inputs=[x1, x2, x3, x4, x5])

    x = GlobalMaxPooling1D()(x)

    dense_1 = Dense(units=512, activation='relu')(x)
    dense_2 = Dense(units=128, activation='relu')(dense_1)
    output_network = Dense(units=predictive_model.number_of_models_involved, activation='softmax')(dense_2)

    predictive_model.architecture = Model(inputs=inputs, outputs=output_network)

def build_wavenet_tcn_classifier_from_encoder_for(predictive_model, input_size):
    inputs = Input(shape=(input_size))
    dense_1 = Dense(units=512, activation='relu')(inputs)
    dense_2 = Dense(units=128, activation='relu')(dense_1)
    output_network = Dense(units=predictive_model.number_of_models_involved, activation='softmax')(dense_2)
    predictive_model.architecture = Model(inputs=inputs, outputs=output_network)

def build_segmentator_for(predictive_model):
    # Networks filters and kernels
    initializer = 'he_normal'
    filters_size = 32
    x1_kernel_size = 4
    x2_kernel_size = 2
    x3_kernel_size = 3
    x4_kernel_size = 10
    x5_kernel_size = 20
    inputs = Input(shape=(predictive_model.trajectory_length, 2))

    x = inputs
    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(x)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=2, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(filters=filters_size, kernel_size=x1_kernel_size, dilation_rate=4, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)
    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(x)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=2, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(filters=filters_size, kernel_size=x2_kernel_size, dilation_rate=4, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling1D()(x2)
    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(x)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=2, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(filters=filters_size, kernel_size=x3_kernel_size, dilation_rate=4, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalAveragePooling1D()(x3)
    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, padding='causal', activation='relu',
                kernel_initializer=initializer)(x)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=4, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(filters=filters_size, kernel_size=x4_kernel_size, dilation_rate=8, padding='causal',
                activation='relu',
                kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = GlobalAveragePooling1D()(x4)
    x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, padding='same', activation='relu',
                kernel_initializer=initializer)(x)
    x5 = BatchNormalization()(x5)
    x5 = GlobalAveragePooling1D()(x5)
    x_concat = concatenate(inputs=[x1, x2, x3, x4, x5])
    dense_1 = Dense(units=(predictive_model.trajectory_length * 2), activation='relu')(x_concat)
    dense_2 = Dense(units=predictive_model.trajectory_length, activation='relu')(dense_1)
    output_network = Dense(units=predictive_model.trajectory_length, activation='sigmoid')(dense_2)

    predictive_model.architecture = Model(inputs=inputs, outputs=output_network)

def build_more_complex_wavenet_tcn_classifier_for(predictive_model, filters=32, number_of_features=2):
    initializer = 'he_normal'
    x1_kernel = 4
    x2_kernel = 2
    x3_kernel = 3
    x4_kernel = 10
    x5_kernel = 6
    x6_kernel = 20

    dilation_depth = 8

    inputs = Input(shape=(predictive_model.trajectory_length-1, number_of_features))

    x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)

    x1 = convolutional_block(predictive_model, x, filters, x1_kernel, [1,2,4], initializer)
    x2 = convolutional_block(predictive_model, x, filters, x2_kernel, [1,2,4], initializer)
    x3 = convolutional_block(predictive_model, x, filters, x3_kernel, [1,2,4], initializer)
    x4 = convolutional_block(predictive_model, x, filters, x4_kernel, [1,4,8], initializer)
    x5 = convolutional_block(predictive_model, x, filters, x5_kernel, [1,2,4], initializer)

    x6 = Conv1D(filters=filters, kernel_size=x6_kernel, padding='same', activation='relu', kernel_initializer=initializer)(x)
    x6 = BatchNormalization()(x6)

    x = concatenate(inputs=[x1, x2, x3, x4, x5, x6])

    x = GlobalMaxPooling1D()(x)

    dense_1 = Dense(units=512, activation='relu')(x)
    dense_2 = Dense(units=128, activation='relu')(dense_1)
    output_network = Dense(units=predictive_model.number_of_models_involved, activation='softmax')(dense_2)

    predictive_model.architecture = Model(inputs=inputs, outputs=output_network)

class ThreadedTrackGenerator(Sequence):
    def __init__(self, batches, batch_size, input_transformer, output_transformer, thread_queue):
        self.batches = batches
        self.batch_size = batch_size
        self.input_transformer = input_transformer
        self.output_transformer = output_transformer
        self.queue = thread_queue

    def __getitem__(self, item):
        trajectories = [self.queue.get() for _ in range(self.batch_size)]
        return self.input_transformer(trajectories), self.output_transformer(trajectories)

    def __len__(self):
        return self.batches

class TrackGenerator(Sequence):
    def __init__(self, batches, batch_size, dataset_function):
        self.batches = batches
        self.batch_size = batch_size
        self.dataset_function = dataset_function

    def __getitem__(self, item):
        tracks, classes = self.dataset_function(self.batch_size)
        return tracks, classes

    def __len__(self):
        return self.batches

#Ploters
def plot_bias(ground_truth, predicted, symbol=None):
    assert symbol in ['alpha', 'd']

    difference = predicted - ground_truth
    x_label = r'$\alpha _{P} - \alpha _{GT}$' if symbol=='alpha' else r'$D _{P} - D _{GT}$'

    sns.kdeplot(difference, color='blue', fill=True)
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Frequency', fontsize=15)
    plt.xlabel(x_label, fontsize=15)

    plt.grid()
    plt.show()

def plot_predicted_and_ground_truth_distribution(ground_truth, predicted):
    sns.kdeplot(ground_truth, color='green', fill=True)
    sns.kdeplot(predicted, color='red', fill=True)
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('Frequency', fontsize=15)
    plt.xlabel('Values', fontsize=15)
    plt.grid()
    plt.show()

def plot_predicted_and_ground_truth_histogram(ground_truth, predicted, a_range=None, title=None, save=False):
    plt.hist2d(ground_truth, predicted, bins=50, range=a_range, cmap=plt.cm.Reds)
    plt.rcParams.update({'font.size': 15})
    plt.xlabel('Ground Truth', fontsize=15)
    plt.ylabel('Predicted', fontsize=15)
    plt.grid()

    if title is not None:
        plt.title(f'{title}')

    if save:
        if title is None:
            title = 'figure'

        plt.savefig(f'{title}.png', dpi=300)
        plt.clf()
    else:
        plt.show()

def get_encoder_from_classifier(a_classifier, layer_index):
    encoding_layer = a_classifier.architecture.layers[layer_index]
    encoding_model = Model(
        inputs=a_classifier.architecture.input,
        outputs=encoding_layer.output
    )

    return encoding_model