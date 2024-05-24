import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, Conv1DTranspose, Dropout, LayerNormalization, MultiHeadAttention, concatenate, Add, Multiply, Layer, GlobalAveragePooling1D, LeakyReLU, Conv2DTranspose, Conv2D, MaxPooling2D, Concatenate, MaxPooling1D
from keras.models import Model

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras import models, Sequential
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import * 
from tensorflow.keras.utils import Sequence


def transform_trajectories_into_turning_angle(predictive_model, trajectories, normalize=True):
    X = np.zeros((len(trajectories), predictive_model.trajectory_length-2, 1))

    for index, trajectory in enumerate(trajectories):
        angles = trajectory.turning_angles(normalized=normalize)

        if len(angles) != 0:
            X[index, :, 0] = trajectory.turning_angles(normalized=normalize)

    return X

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

def transform_trajectories_to_single_level_hurst_exponent(predictive_model, trajectories):
    Y = np.empty((len(trajectories), predictive_model.trajectory_length))

    for index, trajectory in enumerate(trajectories):
        Y[index, :] = trajectory.info['alpha_t']
        Y[index, :] /= 2

    return Y

def transform_trajectories_to_single_level_diffusion_coefficient(predictive_model, trajectories):
    Y = np.empty((len(trajectories), predictive_model.trajectory_length))

    for index, trajectory in enumerate(trajectories):
        Y[index, :] = trajectory.info['d_t']

    return Y

def transform_trajectories_to_single_level_model(predictive_model, trajectories):
    Y = np.empty((len(trajectories), predictive_model.trajectory_length, len(predictive_model.models_involved_in_predictive_model)))

    for index, trajectory in enumerate(trajectories):
        Y[index, :] = to_categorical(trajectory.info['state_t'], num_classes=len(predictive_model.models_involved_in_predictive_model))

    return Y

def transform_trajectories_to_single_level_model_as_number(predictive_model, trajectories):
    Y = np.empty((len(trajectories), predictive_model.trajectory_length))

    for index, trajectory in enumerate(trajectories):
        Y[index, :] = trajectory.info['state_t']

    return Y

def transform_trajectories_to_diffusion_coefficient(predictive_model, trajectories):
    Y = np.empty((len(trajectories), 1))

    for index, trajectory in enumerate(trajectories):
        Y[index, 0] = trajectory.info['diffusion_coefficient']

    return Y

def basic_convolution_block(predictive_model, original_x, filters, kernel_size, dilation_rate, initializer):
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu', kernel_initializer=initializer, dilation_rate=dilation_rate)(original_x)
    x = BatchNormalization()(x)
    return x

def convolutional_block(predictive_model, original_x, filters, kernel_size, dilation_rates, initializer, activation='relu'):
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation=activation, kernel_initializer=initializer, dilation_rate=dilation_rates[0])(original_x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rates[1], padding='causal', activation=activation, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rates[2], padding='causal', activation=activation, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation=activation, kernel_initializer=initializer)(original_x)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])

    return x

#https://www.tensorflow.org/text/tutorials/transformer#the_feed_forward_network
class BaseAttention(Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = MultiHeadAttention(**kwargs)
    self.layernorm = LayerNormalization()
    self.add = Add()

#https://www.tensorflow.org/text/tutorials/transformer#the_feed_forward_network
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

#https://www.tensorflow.org/text/tutorials/transformer#the_feed_forward_network
class EncoderLayer(Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

#https://www.tensorflow.org/text/tutorials/transformer#the_feed_forward_network
class FeedForward(Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = Sequential([
      Dense(dff, activation='relu'),
      Dense(d_model),
      Dropout(dropout_rate)
    ])
    self.add = Add()
    self.layer_norm = LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
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

def build_wavenet_tcn_segmenter_from_encoder_for(predictive_model, input_size):
    inputs = Input(shape=(input_size))
    dense_1 = Dense(units=(predictive_model.trajectory_length * 2), activation='relu')(inputs)
    dense_2 = Dense(units=predictive_model.trajectory_length, activation='relu')(dense_1)
    output_network = Dense(units=predictive_model.trajectory_length, activation='sigmoid')(dense_2)
    predictive_model.architecture = Model(inputs=inputs, outputs=output_network)

def build_segmentator_for(predictive_model, with_wadnet=False, number_of_features=2, filters=32, input_size=None, with_skip_connections=False):

    if input_size is None:
        input_size = predictive_model.trajectory_length

    # Networks filters and kernels
    initializer = 'he_normal'
    filters_size = filters
    x1_kernel_size = 4
    x2_kernel_size = 2
    x3_kernel_size = 3
    x4_kernel_size = 10
    x5_kernel_size = 20
    inputs = Input(shape=(input_size, number_of_features))

    if with_wadnet:
        x = WaveNetEncoder(filters_size, 8, initializer=initializer)(inputs)
    else:
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

    if with_skip_connections:
        x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x_skip = BatchNormalization()(x_skip)
        x1 = Add()([x1, x_skip])

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

    if with_skip_connections:
        x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x_skip = BatchNormalization()(x_skip)
        x2 = Add()([x2, x_skip])

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

    if with_skip_connections:
        x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x_skip = BatchNormalization()(x_skip)
        x3 = Add()([x3, x_skip])

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

    if with_skip_connections:
        x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x_skip = BatchNormalization()(x_skip)
        x4 = Add()([x4, x_skip])

    x4 = GlobalAveragePooling1D()(x4)
    x5 = Conv1D(filters=filters_size, kernel_size=x5_kernel_size, padding='same', activation='relu',
                kernel_initializer=initializer)(x)
    x5 = BatchNormalization()(x5)

    if with_skip_connections:
        x_skip = Conv1D(filters=filters, kernel_size=1, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x_skip = BatchNormalization()(x_skip)
        x5 = Add()([x5, x_skip])

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

def get_target_image(image_of_particles, circle_radius):
    target_image = np.zeros(image_of_particles.shape)
    X, Y = np.meshgrid(
        np.arange(0, image_of_particles.shape[0]), 
        np.arange(0, image_of_particles.shape[1])
    )

    for property in image_of_particles.properties:
        if "position" in property:
            position = property["position"]

            distance_map = (X - position[1])**2 + (Y - position[0])**2
            target_image[distance_map < circle_radius**2] = 1
    
    return target_image

class ImageGenerator(Sequence):
    def __init__(self, batches, batch_size, image_width, image_height, circle_radius, deeptrack_feature):
        self.batches = batches
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.circle_radius = circle_radius
        self.deeptrack_feature = deeptrack_feature

    def __getitem__(self, item):        
        X = np.zeros((self.batch_size,self.image_width,self.image_height,1))
        Y = np.zeros((self.batch_size,self.image_width,self.image_height,1))

        for i in range(self.batch_size):
            self.deeptrack_feature.update()
            image_of_particles =  self.deeptrack_feature.resolve()        
            X[i] = image_of_particles/255
            Y[i] = get_target_image(image_of_particles, self.circle_radius)

        return X, Y

    def __len__(self):
        return self.batches

"""
This code functions were based from 
https://github.com/Nguyendat-bit/U-net/tree/main
which were used to repeat the buggy architecture coded from 
https://github.com/DeepTrackAI/DeepTrack2/blob/develop/examples/paper-examples/4-multi-molecule-tracking.ipynb
"""
def down_block(x, filters, use_maxpool = True, input_dimension='2d', basic_kernel_size=3):
    if input_dimension=='2d':
        x = Conv2D(filters, basic_kernel_size, padding= 'same')(x)
    elif input_dimension=='1d':
        x = Conv1D(filters, basic_kernel_size, padding= 'causal')(x)

    x = LeakyReLU()(x)
    #x = BatchNormalization()(x)
    if use_maxpool == True:
        if input_dimension=='2d':
            pooling = MaxPooling2D(strides= (2,2))(x)
        elif input_dimension=='1d':
            pooling = MaxPooling1D(strides= 2)(x)
        return  pooling, x
    else:
        return x

def up_block(x,y, filters, input_dimension='2d', basic_kernel_size=3):
    if input_dimension=='2d':
        x = Conv2D(filters, basic_kernel_size, padding= 'same')(x)
    elif input_dimension=='1d':
        x = Conv1D(filters, basic_kernel_size, padding= 'causal')(x)

    x = LeakyReLU()(x)
    #x = BatchNormalization()(x)
    if input_dimension=='2d':
        x = Conv2DTranspose(filters//2, (2,2), strides=2)(x)
    elif input_dimension=='1d':
        x = Conv1DTranspose(filters//2, 2, strides=2)(x)

    if input_dimension=='2d':
        axis_dimension = 3
    elif input_dimension=='1d':
        axis_dimension = 2

    x = Concatenate(axis = axis_dimension)([x,y])
    return x
    
def Unet(input_size, input_dimension='2d', basic_kernel_size=3, unet_index=None, skip_last_block=False):
    input = Input(shape = input_size)
    x, temp1 = down_block(input, 16, input_dimension=input_dimension, basic_kernel_size=basic_kernel_size)
    x, temp2 = down_block(x, 32, input_dimension=input_dimension, basic_kernel_size=basic_kernel_size)
    x, temp3 = down_block(x, 64, input_dimension=input_dimension, basic_kernel_size=basic_kernel_size)
    x = down_block(x, 128, use_maxpool= False, input_dimension=input_dimension, basic_kernel_size=basic_kernel_size)

    x = up_block(x,temp3, 128, input_dimension=input_dimension, basic_kernel_size=basic_kernel_size)
    x = up_block(x,temp2, 64, input_dimension=input_dimension, basic_kernel_size=basic_kernel_size)
    x = up_block(x,temp1, 32, input_dimension=input_dimension, basic_kernel_size=basic_kernel_size)

    if input_dimension=='2d':
        x = Conv2D(16, basic_kernel_size, padding= 'same')(x)
    elif input_dimension=='1d':
        x = Conv1D(16, basic_kernel_size, padding= 'causal')(x)

    x = LeakyReLU()(x)
    #x = BatchNormalization()(x)
    if input_dimension=='2d':
        x = Conv2D(16, basic_kernel_size, padding= 'same')(x)
    elif input_dimension=='1d':
        x = Conv1D(16, basic_kernel_size, padding= 'causal')(x)

    x = LeakyReLU()(x)
    #x = BatchNormalization()(x)
    if input_dimension=='2d':
        x = Conv2D(16, basic_kernel_size, padding= 'same')(x)
    elif input_dimension=='1d':
        x = Conv1D(16, basic_kernel_size, padding= 'causal')(x)

    x = LeakyReLU()(x)
    #x = BatchNormalization()(x)

    if not skip_last_block:
        if input_dimension=='2d':
            output = Conv2D(1, basic_kernel_size, activation= 'sigmoid', padding='same')(x)
        elif input_dimension=='1d':
            output = Conv1D(1, basic_kernel_size, activation= 'sigmoid', padding= 'causal')(x)
    else:
        output = x
    model = models.Model(input, output, name = f'unet_{unet_index}' if unet_index is not None else 'unet')
    return model

#Ploters
def plot_bias(ground_truth, predicted, symbol=None, a_range=None, file_name=None):
    assert symbol in ['alpha', 'd']

    difference = predicted - ground_truth
    x_label = r'$\alpha _{P} - \alpha _{GT}$' if symbol=='alpha' else r'$D _{P} - D _{GT}$'

    sns.kdeplot(difference, color='blue', fill=True)
    plt.rcParams.update({'font.size': 15})
    plt.ylabel('', fontsize=30)
    plt.xlabel(x_label, fontsize=30)

    plt.yticks([])
    
    if a_range is not None:
        plt.xticks([a_range[0],0,a_range[1]], fontsize=30)
        plt.xlim(a_range)

    plt.grid(color='black')
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)
        plt.clf()
    else:
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