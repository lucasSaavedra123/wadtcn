import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import device, config
import deeptrack as dt
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import math

from .PredictiveModel import PredictiveModel
from CONSTANTS import *
from .model_utils import ImageGenerator, Unet
from utils import create_trajectories
from Trajectory import Trajectory


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

    def predict(
            self,
            image_array,
            extract_trajectories=False,
            extract_localizations=False,
            pixel_size=100e-9,
            classification_threshold = 0.5,
            spt_max_distance_tolerance = 1000e-9,
            debug=False,
            plot_trajectories=False,
        ):

        unet_result = (self.architecture.predict(image_array/255)[...,0] > classification_threshold).astype(int)

        if not extract_localizations:
            return unet_result

        data = []

        #Code from https://github.com/DeepTrackAI/DeepTrack2/blob/develop/examples/paper-examples/4-multi-molecule-tracking.ipynb
        for frame_index, (mask, frame) in enumerate(zip(unet_result, image_array)):
            raw_localizations = []

            if debug:
                plt.title(f"Frame {frame_index}")
                plt.imshow(frame)
                plt.show()
            
            cs = skimage.measure.regionprops(skimage.measure.label(mask))
            raw_localizations = [list(c["Centroid"])[::-1] for c in cs if c.axis_major_length < 8]

            for props in [ci for ci in cs if ci.axis_major_length >= 8]:
                y0, x0 = props.centroid
                orientation = props.orientation

                x_bottom = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
                y_bottom = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
                x_top = x0 + math.sin(orientation) * 0.5 * props.axis_major_length
                y_top = y0 + math.cos(orientation) * 0.5 * props.axis_major_length

                x_offset = (x_top - x_bottom) / 4
                y_offset = (y_top - y_bottom) / 4

                new_x_1 = x_top-x_offset
                new_x_2 = x_bottom+x_offset

                new_y_1 = y_top-y_offset
                new_y_2 = y_bottom+y_offset

                raw_localizations += [[new_x_1, new_y_1], [new_x_2, new_y_2]]

            data += [[frame_index]+p for p in raw_localizations]
            raw_localizations = np.array(raw_localizations)

            if debug:
                plt.title(f"Localizations from Frame Index: {frame_index}")
                plt.imshow(frame)
                plt.scatter(raw_localizations[:,0], raw_localizations[:,1], marker='X', color='red')
                plt.show()

        """
        Localizations are multiplied by the pixel size.
        The origin of the axis is positioned to the 
        left-bottom corner of the movie.
        """
        data = np.array(data)
        data[:,1:] *= pixel_size
        data[:,2] *= -1
        data[:,2] += image_array.shape[2] * pixel_size

        if not extract_trajectories:
            return pd.DataFrame({
                'frame': data[:,0],
                'x': data[:,1],
                'y': data[:,2],
            })

        #Code From https://drive.google.com/drive/u/0/folders/1lOKvC_L2fb78--uwz3on4lBzDGVum8Mc
        tracked_data = np.column_stack((data, np.zeros((len(data),3))))
        tracksCounter = 1

        for fr in range(0, int(np.max(tracked_data[:,0]))):
            [tracked_data,tracksCounter] = create_trajectories(tracked_data,fr,fr+1,spt_max_distance_tolerance,tracksCounter)

        dataframe = pd.DataFrame({
            'frame':tracked_data[:,0],
            'x': tracked_data[:,1],
            'y':tracked_data[:,2],
            'track_id': tracked_data[:,4],
        })

        track_ids = np.unique(tracked_data[:,4])
        trajectories = []

        for track_id in track_ids:
            track_dataframe = dataframe[dataframe['track_id'] == track_id].sort_values('frame')
            
            new_trajectory = Trajectory(
                x=track_dataframe['x'].tolist(),
                y=track_dataframe['y'].tolist(),
                t=track_dataframe['frame'].tolist(),
                noisy=True
            )

            if plot_trajectories:
                plt.plot(new_trajectory.get_noisy_x(), new_trajectory.get_noisy_y(), marker='X')

            trajectories.append(new_trajectory)

        if plot_trajectories:
            plt.show()

        return trajectories

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
