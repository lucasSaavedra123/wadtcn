import numpy as np
from keras.layers import Conv1D, BatchNormalization, Add


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

    return X

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
