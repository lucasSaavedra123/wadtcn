import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import device, config
import deeptrack as dt
import tensorflow
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.feature.peak import peak_local_max

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
            gaussian_filter_sigma=2,
            pixel_size=100e-9,
            peak_local_max_min_distance = 3,
            peak_fitting_roi_radius = 5,
            spt_max_distance_tolerance = 1000e-9,
            debug=False,
            plot_trajectories=False,
            extract_blured_movie=False
        ):

        unet_result = (self.architecture.predict(image_array/255)[...,0] > 0.5).astype(int)

        if not extract_trajectories:
            return unet_result
        
        predicted_movie = unet_result*255
        blured_movie = gaussian_filter(predicted_movie, gaussian_filter_sigma)

        if extract_blured_movie:
            return blured_movie

        data = []

        #Code From https://drive.google.com/drive/u/0/folders/1lOKvC_L2fb78--uwz3on4lBzDGVum8Mc
        for frame_index, blured_frame in enumerate(blured_movie):
            peaks = peak_local_max(blured_frame, peak_local_max_min_distance, threshold_abs = np.std(blured_frame)*2)

            raw_localizations = []

            if debug:
                plt.title(f"Peaks from Frame Index: {frame_index}")
                plt.imshow(blured_frame)
                plt.scatter(peaks[:,1], peaks[:,0])
                plt.show()

            for peak in peaks:
                ROI = blured_frame[
                    peak[0]-peak_fitting_roi_radius:peak[0]+peak_fitting_roi_radius+1,
                    peak[1]-peak_fitting_roi_radius:peak[1]+peak_fitting_roi_radius+1
                ]

                if 0 in ROI.shape:
                    continue
                
                ROI_F = np.fft.fft2(ROI)

                xangle = np.arctan(ROI_F[0,1].imag/ROI_F[0,1].real) - np.pi
                if xangle > 0:
                    xangle -= 2*np.pi
                PositionX = abs(xangle)/(2*np.pi/(peak_fitting_roi_radius*2+1))

                yangle = np.arctan(ROI_F[1,0].imag/ROI_F[1,0].real) - np.pi
                if yangle > 0:
                    yangle -= 2*np.pi
                PositionY = abs(yangle)/(2*np.pi/(peak_fitting_roi_radius*2+1))

                new_x = peak[1]-peak_fitting_roi_radius+PositionX
                new_y = peak[0]-peak_fitting_roi_radius+PositionY

                data.append([frame_index, new_x, new_y])
                raw_localizations.append([new_x, new_y])

            raw_localizations = np.array(raw_localizations)

            if debug:
                plt.title(f"Localizations from Frame Index: {frame_index}")
                plt.imshow(blured_frame)
                plt.scatter(raw_localizations[:,0], raw_localizations[:,1])
                plt.show()

        tracked_data = np.column_stack((data, np.zeros((len(data),3))))
        tracked_data[:,1:3] *= pixel_size
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
