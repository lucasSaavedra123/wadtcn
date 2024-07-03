from matplotlib import pyplot as plt
from PredictiveModel.UNetSingleParticleTracker import UNetSingleParticleTracker
from utils import tiff_movie_path_to_numpy_array
import numpy as np
import cv2
from time import sleep
from scipy.ndimage import gaussian_filter
from IPython import embed


def get_id_of_position(a_dict, position):
    for an_id in a_dict:
        x_tuple = a_dict[an_id]['x']
        y_tuple = a_dict[an_id]['y']
        if x_tuple[0]<position[0]<x_tuple[1] and y_tuple[0]<position[1]<y_tuple[1]:
            return an_id
    return False

def conv(img, size):
    mask = np.ones([size, size], dtype = int) 
    mask = mask / (size*size)

    m, n = img.shape

    # Convolve the 3X3 mask over the image  
    img_new = np.zeros([m, n]) 
    
    for i in range(1, m-1): 
        for j in range(1, n-1): 
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2] 
            
            img_new[i, j]= temp 
            
    img_new = img_new.astype(np.uint8)
    return img_new

#https://github.com/GanzingerLab/SPIT check that code please
network = UNetSingleParticleTracker(128,128,2)
network.load_as_file()

movie = tiff_movie_path_to_numpy_array('public_data_challenge_v0/track_1/exp_6/videos_fov_2.tiff')
mask = movie[0]
id_to_pixel_position = {}
ids = np.unique(mask).tolist()
ids.remove(255)

for an_id in ids:
    y_position, x_position = np.where(mask == an_id)

    id_to_pixel_position[an_id] = {
        'x': (np.min(x_position), np.max(x_position)),
        'y': (np.min(y_position), np.max(y_position)),
    }

movie = movie[1:]
processed_movie = movie.copy()

for frame_i in range(processed_movie.shape[0]):
    processed_movie[frame_i] = conv(processed_movie[frame_i],3)
    #processed_movie[frame_i] = conv(processed_movie[frame_i],5)
    #processed_movie[frame_i] = gaussian_filter(processed_movie[frame_i], 3)

dataframe = network.predict(processed_movie, pixel_size=1, extract_trajectories_as_dataframe=True, spt_max_distance_tolerance=15)
dataframe['vip'] = False

first_frame_dataframe = dataframe[dataframe['frame']==0].copy()

for track_id in first_frame_dataframe['track_id'].unique():
    track = first_frame_dataframe[first_frame_dataframe['track_id']==track_id]
    if get_id_of_position(id_to_pixel_position, track[['x','y']].values[0]) != False:
        dataframe.loc[dataframe['track_id']==track_id, 'vip'] = True

movie = np.repeat(movie[..., np.newaxis], 3, axis=-1)
i = 0

while True: 
    i += 1
    frame_index = i%30

    frame = movie[frame_index]

    for row_index, row in dataframe[dataframe['frame'] == frame_index].iterrows():
        if row.vip:
            frame = cv2.circle(frame, (int(row.x.real),int(row.y.real)), radius=1, color=(255,0,0))
        else:
            frame = cv2.circle(frame, (int(row.x.real),int(row.y.real)), radius=1, color=(0,0,255))

    up_to_frame = dataframe[dataframe['frame'] <= frame_index]
    
    for track_id in up_to_frame['track_id'].unique():
        track = up_to_frame[up_to_frame['track_id']==track_id].sort_values('frame')
        draw_points = (np.asarray([track['x'].values, track['y'].values]).T).astype(np.int32)   # needs to be int32 and transposed
        frame = cv2.polylines(frame, [draw_points], False, (0,0,0))

    #for an_id in id_to_pixel_position:
    #    frame = cv2.circle(frame, (int(id_to_pixel_position[an_id]['x'][0]),int(id_to_pixel_position[an_id]['y'][0])), radius=1, color=(255,255,0))

    frame = cv2.resize(frame, (1024//2, 1024//2))
    # this function will concatenate 
    # the two matrices 
    cv2.imshow('animation', frame) 
  
    if cv2.waitKey(1) == ord('q'): 
        # press q to terminate the loop 
        cv2.destroyAllWindows() 
        break
    
    #sleep(0.5)