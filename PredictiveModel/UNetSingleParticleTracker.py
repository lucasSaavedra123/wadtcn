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
            'epochs': 10
        }

    def default_hyperparameters(self):
        return {
            'lr': 0.001,
            'batch_size': 32,
            'amsgrad': False,
            'epsilon': 1e-8,
            'epochs': 10
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
        return (self.architecture.predict(image_array/255)[...,0] > 0.5).astype(int)

    @property
    def type_name(self):
        return "unet_single_particle_tracker"

    def __str__(self):
        return f"{self.type_name}"

    def fit(self):
        if not self.trained:
            self.build_network()
            real_epochs = self.hyperparameters['epochs']
        else:
            real_epochs = self.hyperparameters['epochs'] - len(self.history_training_info['loss'])

        _particle_dict = {
            "particle_intensity": [500, 20,],  # Mean and standard deviation of the particle intensity
            "intensity": lambda particle_intensity: particle_intensity[0]
            + np.random.randn() * particle_intensity[1],
            "intensity_variation": 0,  # Intensity variation of particle (in standard deviation)
            "z": None, # Placeholder for z
            "refractive_index": 1.45,  # Refractive index of the particle
            "position_unit": "pixel",
        }

        _optics_dict = {
            "NA": 1.46,  # Numerical aperture
            "wavelength": 500e-9,  # Wavelength
            "resolution": 100e-9,  # Camera resolution or effective resolution
            "magnification": 1,
            "refractive_index_medium": 1.33,
            "output_region": [0, 0, 128, 128],
        }

        # Background offset
        _background_dict = {
            "background_mean": 100,  # Mean background intensity
            "background_std": 0,  # Standard deviation of background intensity within a video
        }

        # Generate point particles
        particle = dt.PointParticle(
            position=lambda: np.random.rand(2) * IMAGE_SIZE,
            **{k: v for k, v in _particle_dict.items() if k != 'z'},
        )

        # Adding background offset
        background = dt.Add(
            value=_background_dict["background_mean"]
            + np.random.randn() * _background_dict["background_std"]
        )

        # Define optical setup
        optics = dt.Fluorescence(**_optics_dict)

        # Normalising image plane particle intensity
        scale_factor = (
            (
                optics.magnification()
                * optics.wavelength()
                / (optics.NA() * optics.resolution())
            )
            ** 2
        ) * (1 / np.pi)
        scale_factor = 4 * np.sqrt(scale_factor) # Scaling to the peak value

        # Poisson noise
        def func_poisson_noise():
            """
            Applies poisson noise to an image.

            This is a custom DeepTrack feature, and a helper function for `transform_to_video`
            """
            def inner(image):
                image[image<0] = 0
                rescale = 1
                noisy_image = np.random.poisson(image * rescale) / rescale
                return noisy_image
            return inner

        poisson_noise = dt.Lambda(func_poisson_noise)
        num_particles = lambda: np.random.randint(1, 20)

        def func_convert_to_uint8():
            def inner(image):
                image = image / image.max()
                image = image * 255
                image = image.astype(np.uint8)
                return image

            return inner

        convert_to_uint8 = dt.Lambda(func_convert_to_uint8)

        self.image_features = (
            optics(particle ^ num_particles)
            >> dt.Multiply(scale_factor)
            >> background
            >> poisson_noise
            >> convert_to_uint8
        )

        self.build_network()

        self.architecture.summary()

        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        train_set_size = 10_000
        val_set_size = 1_000

        with device(device_name):
            history_training_info = self.architecture.fit(
                ImageGenerator(train_set_size//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.image_features),
                epochs=real_epochs,
                validation_data=ImageGenerator(val_set_size//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.image_features),
                shuffle=True
            ).history

        if self.trained:
            for dict_key in history_training_info:
                self.history_training_info[dict_key] += history_training_info[dict_key]
        else:
            self.history_training_info = history_training_info
            self.trained = True
