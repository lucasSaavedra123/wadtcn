from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam

from TheoreticalModels.FractionalBrownianMotion import FractionalBrownianMotionBrownian, FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionSuperDiffusive
from .PredictiveModel import PredictiveModel
from .model_utils import convolutional_block, WaveNetEncoder, transform_trajectories_to_diffusion_coefficient, transform_trajectories_into_squared_differences

class WavenetTCNWithLSTMDiffusionCoefficientPredicter(PredictiveModel):
    #These will be updated after hyperparameter search

    def model_string_to_class_dictionary(self):
        return {
            'fbm_sub': { 'model': FractionalBrownianMotionSubDiffusive },
            'fbm_brownian': { 'model': FractionalBrownianMotionBrownian },
            'fbm_sup': { 'model': FractionalBrownianMotionSuperDiffusive },
        }

    def default_hyperparameters(self, **kwargs):
        model_string_to_hyperparameters_dictionary = {
            'fbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'fbm_brownian': {'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 100},
            'fbm_sup': {'lr': 0.0001, 'batch_size': 8, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[kwargs["model"]]
        hyperparameters['epochs'] = 100

        return hyperparameters

    @classmethod
    def selected_hyperparameters(self, model_label):
        model_string_to_hyperparameters_dictionary = {
            'fbm_sub': {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
            'fbm_brownian': {'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 100},
            'fbm_sup': {'lr': 0.0001, 'batch_size': 8, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 100},
        }

        hyperparameters = model_string_to_hyperparameters_dictionary[model_label]
        hyperparameters['epochs'] = 100

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
        return [self.model_string_to_class_dictionary()[self.extra_parameters["model"]]['model']]

    def predict(self, trajectories):
        return self.architecture.predict(self.transform_trajectories_to_input(trajectories))

    def transform_trajectories_to_output(self, trajectories):
        return transform_trajectories_to_diffusion_coefficient(self, trajectories)

    def transform_trajectories_to_input(self, trajectories):
        return transform_trajectories_into_squared_differences(self, trajectories, normalize=True)

    def build_network(self):
        inputs = Input(shape=(self.trajectory_length-1, 1))
        filters = 32
        dilation_depth = 8
        initializer = 'he_normal'

        x = WaveNetEncoder(filters, dilation_depth, initializer=initializer)(inputs)
        x = convolutional_block(self, x, filters, 3, [1,2,4], initializer)
        x = GlobalAveragePooling1D()(x)

        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=128, activation='relu')(x)
        output_network = Dense(units=1, activation='sigmoid')(x)

        self.architecture = Model(inputs=inputs, outputs=output_network)

        optimizer = Adam(
            lr=self.hyperparameters['lr'],
            epsilon=self.hyperparameters['epsilon'],
            amsgrad=self.hyperparameters['amsgrad']
        )

        self.architecture.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    @property
    def type_name(self):
        return 'wavenet_diffusion_coefficient'
