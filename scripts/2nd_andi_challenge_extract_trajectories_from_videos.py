from os.path import join
from os import makedirs
from collections import defaultdict

from IPython import embed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import tqdm
from andi_datasets.utils_videos import play_video
import matplotlib.animation as animation
import moviepy.editor as mp

from Trajectory import Trajectory
from PredictiveModel.UNetSingleParticleTracker import UNetSingleParticleTracker
from utils import tiff_movie_path_to_numpy_array

PUBLIC_DATA_PATH = './public_data_validation_v1'
PATH_TRACK_1 = './track_1'

N_EXP = 10
N_FOVS = 30

unet_network = UNetSingleParticleTracker(128,128,2)
unet_network.load_as_file()

#All trajectories are extracted from videos and saved for further inference
trajectories = []

for exp in tqdm.tqdm(list(range(N_EXP))):
    for fov in range(N_FOVS):
        tiff_file_path = join(PUBLIC_DATA_PATH, PATH_TRACK_1, f'exp_{exp}', f'videos_fov_{fov}.tiff')
        video = tiff_movie_path_to_numpy_array(tiff_file_path)

        mask = video[0]
        video = video[1:]

        result = unet_network.predict(video, extract_trajectories=True, pixel_size=1, spt_max_distance_tolerance=7, plot_trajectories=True)
        #plt.figure(figsize=(5, 5))
        #plt.imshow(video[0], cmap="gray", zorder = -1)
        #for traj in result:
        #    plt.plot(traj.get_noisy_x(), traj.get_noisy_y(), alpha=0.2)
        #plt.xlim(0,128); plt.ylim(0,128)
        #plt.show()
        """
        fig, ax = plt.subplots()
        frame_Dataframe = result[result['frame']==0]
        scat = ax.scatter(frame_Dataframe['x'], frame_Dataframe['y'],color='red', alpha=0.6)
        img = ax.imshow(video[0])
        ax.set(xlim=[0,128], ylim=[0,128])

        def update(frame):
            frame_Dataframe = result[result['frame']==frame]
            scat.set_offsets(frame_Dataframe[['x','y']].values)
            img.set_data(video[frame])
            plt.tight_layout()
            return (img, scat)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=video.shape[0], interval=1)
        #plt.show()
        ani.save(f'DELETE.gif', writer=animation.PillowWriter(fps=5), dpi=300)
        clip = mp.VideoFileClip(f'DELETE.gif')
        clip.write_videofile(f'./{exp}_{fov}.mp4')
        exit()
        """