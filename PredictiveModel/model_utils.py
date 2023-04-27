import numpy as np
from keras.layers import Conv1D, BatchNormalization, Add, Layer, Multiply
from tensorflow.keras.utils import to_categorical


def transform_trajectories_into_displacements(predictive_model, trajectories):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length-1, 2))

    def axis_adaptation_to_net(axis_data, track_length):
        axis_reshaped = np.reshape(axis_data, newshape=[1, len(axis_data)])
        axis_reshaped = axis_reshaped - np.mean(axis_reshaped)
        axis_diff = np.diff(axis_reshaped[0, :track_length])
        return axis_diff

    for index, trajectory in enumerate(trajectories):
        X[index, :, 0] = axis_adaptation_to_net(trajectory.get_noisy_x(), predictive_model.trajectory_length)
        X[index, :, 1] = axis_adaptation_to_net(trajectory.get_noisy_y(), predictive_model.trajectory_length)

        if predictive_model.simulator().STRING_LABEL == 'andi':
            X[index, :, 0] = (X[index, :, 0] - np.mean(X[index, :, 0]))/(np.std(X[index, :, 0]) if np.std(X[index, :, 0])!= 0 else 1)
            X[index, :, 1] = (X[index, :, 1] - np.mean(X[index, :, 1]))/(np.std(X[index, :, 1]) if np.std(X[index, :, 1])!= 0 else 1)

    return X

def transform_trajectories_into_raw_trajectories(predictive_model, trajectories):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length, 2))

    for index, trajectory in enumerate(trajectories):
        X[index, :, 0] = trajectory.get_noisy_x() - np.mean(trajectory.get_noisy_x())
        X[index, :, 1] = trajectory.get_noisy_y() - np.mean(trajectory.get_noisy_y())

        if predictive_model.simulator().STRING_LABEL == 'andi':
            X[index, :, 0] = X[index, :, 0]/(np.std(X[index, :, 0]) if np.std(X[index, :, 0])!= 0 else 1)
            X[index, :, 1] = X[index, :, 1]/(np.std(X[index, :, 1]) if np.std(X[index, :, 1])!= 0 else 1)

    return X

def transform_trajectories_to_categorical_vector(predictive_model, trajectories):
    Y_as_vectors = np.empty((len(trajectories), predictive_model.number_of_models_involved))

    for index, trajectory in enumerate(trajectories):
        Y_as_vectors[index, :] = to_categorical(predictive_model.model_to_label(trajectory.model_category), num_classes=predictive_model.number_of_models_involved)

    return Y_as_vectors

def transform_trajectories_to_hurst_exponent(predictive_model, trajectories):
    Y = np.empty((len(trajectories), 1))

    for index, trajectory in enumerate(trajectories):
        Y[index, 0] = trajectory.hurst_exponent()

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