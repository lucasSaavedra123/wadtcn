import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import device, config
import deeptrack as dt
import tensorflow


from .PredictiveModel import PredictiveModel
from CONSTANTS import *
from .model_utils import ImageGenerator, Unet


class UNetSingleParticleTracker(PredictiveModel):
    @classmethod
    def selected_hyperparameters(self):
        return {
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 100
        }

    def default_hyperparameters(self):
        return {
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 100
        }

    @classmethod
    def default_hyperparameters_analysis(self):
        pass

    def __init__(self, **kwargs):
        self.architecture = None
        self.hyperparameters_analysis = self.__class__.default_hyperparameters_analysis()
        self.db_persistance = False
        self.early_stopping = False
        self.wadnet_tcn_encoder = None

        if 'hyperparameters' in kwargs:
            hyperparameters = kwargs['hyperparameters']
            del kwargs['hyperparameters']
        else:
            hyperparameters = self.default_hyperparameters(**kwargs)

        if 'id' in kwargs:
            super().__init__(
                trajectory_length=None,
                trajectory_time=None,
                hyperparameters=hyperparameters,
                simulator_identifier=None,
                **kwargs
            )
        else:
            super().__init__(
                trajectory_length=None,
                trajectory_time=None,
                hyperparameters=hyperparameters,
                simulator_identifier=None,
                extra_parameters = kwargs
            )

    def build_network(self):
        self.architecture = Unet((IMAGE_SIZE,IMAGE_SIZE,1))

        loss = dt.losses.flatten(
            dt.losses.weighted_crossentropy((10, 1))
        )
        metric = dt.losses.flatten(
            dt.losses.weighted_crossentropy((1, 1))
        )

        optimizer = Adam(
            lr=self.hyperparameters['lr'],
            amsgrad=self.hyperparameters['amsgrad'],
            epsilon=self.hyperparameters['epsilon']
        )

        self.architecture.compile(optimizer= optimizer, loss=loss, metrics=[metric])

    def predict(self, image_array):
        return self.architecture.predict(image_array)

    @property
    def type_name(self):
        return "unet_single_particle_tracker"

    def __str__(self):
        f"{self.type_name}"

    def fit(self):
        if not self.trained:
            self.build_network()
            real_epochs = self.hyperparameters['epochs']
        else:
            real_epochs = self.hyperparameters['epochs'] - len(self.history_training_info['loss'])

        particle = dt.PointParticle(                                         
            intensity=100,
            position=lambda: np.random.rand(2) * IMAGE_SIZE
        )

        fluorescence_microscope = dt.Fluorescence(
            NA=0.7,                
            resolution=1e-6,     
            magnification=10,
            wavelength=680e-9,
            output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE)
        )

        offset = dt.Add(
            value=lambda: np.random.rand()*1
        )

        poisson_noise = dt.Poisson(
            snr=lambda: np.random.rand()*7 + 3,
            background=offset.value
        )

        num_particles = lambda: np.random.randint(1, 11)

        self.image_features = fluorescence_microscope(particle^num_particles) >> offset >> poisson_noise

        self.build_network()

        self.architecture.summary()

        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        with device(device_name):
            history_training_info = self.architecture.fit(
                ImageGenerator(TRAINING_SET_SIZE_PER_EPOCH//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.image_features),
                epochs=real_epochs,
                validation_data=ImageGenerator(VALIDATION_SET_SIZE_PER_EPOCH//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.image_features),
                shuffle=True
            ).history

        if self.trained:
            for dict_key in history_training_info:
                self.history_training_info[dict_key] += history_training_info[dict_key]
        else:
            self.history_training_info = history_training_info
            self.trained = True
