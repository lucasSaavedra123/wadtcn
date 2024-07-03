from time import sleep
from collections import defaultdict

import numpy as np
import cv2

from PredictiveModel.UNetSingleParticleTracker import UNetSingleParticleTracker
from utils import tiff_movie_path_to_numpy_array, get_trajectories_from_2nd_andi_challenge_tiff_movie

DATA_PATH = './public_data_challenge_v0'
EXP = 5
FOV = 3

network = UNetSingleParticleTracker(128,128,2)
network.load_as_file()

movie = tiff_movie_path_to_numpy_array(f'{DATA_PATH}/track_1/exp_{EXP}/videos_fov_{FOV}.tiff')
dataframe = get_trajectories_from_2nd_andi_challenge_tiff_movie(movie,network)

movie = movie[1:]
movie = np.repeat(movie[..., np.newaxis], 3, axis=-1)

i = 0
while True:
    i += 1
    frame_index = i%len(movie)

    frame = movie[frame_index]

    for row_index, row in dataframe[dataframe['frame'] == frame_index].iterrows():
        color = (255,0,255) if row.vip else (0,0,255)
        frame = cv2.circle(frame, (int(row.x.real),int(row.y.real)), radius=1, color=color)

    up_to_frame = dataframe[dataframe['frame'] <= frame_index]
    
    for track_id in up_to_frame['traj_idx'].unique():
        track = up_to_frame[up_to_frame['traj_idx']==track_id].sort_values('frame')
        draw_points = (np.asarray([track['x'].values, track['y'].values]).T).astype(np.int32)
        frame = cv2.polylines(frame, [draw_points], False, (0,0,0))

    #for an_id in id_to_pixel_position:
    #    frame = cv2.circle(frame, (int(id_to_pixel_position[an_id]['x'][0]),int(id_to_pixel_position[an_id]['y'][0])), radius=1, color=(255,255,0))

    frame = cv2.resize(frame, (1024//2, 1024//2))
    cv2.imshow(f'Data path:{DATA_PATH}, Experiment:{EXP}, FOV:{FOV}', frame) 
    if cv2.waitKey(1) == ord('q'): 
        cv2.destroyAllWindows() 
        break
    sleep(0.10)
