import numpy as np
from tifffile import TiffWriter, TiffFile
from collections import defaultdict

import AutoStepFinder.stepfindCore as core
import AutoStepFinder.stepfindTools as st


def tiff_movie_path_to_numpy_array(tiff_movie_path):
    frames = []
    
    with TiffFile(tiff_movie_path) as tif:
        for page in tif.pages:
            frames.append(page.asarray())

    return np.stack(frames)

def numpy_array_to_tiff_movie_path(numpy_array, tiff_movie_path):
    with TiffWriter(tiff_movie_path) as tif:
        for frame in numpy_array:
            tif.write(frame, contiguous=True)

"""
Yo considero como 'espurio' a aquellos breakpoints consecutivos 
que se dan a una distancia muy cerca entre si. La idea es 
reemplazarlos por uno que actue como 'promedio' entre ambos
"""
def merge_spurious_break_points_by_distance(a_list, distance):
    new_list = []
    i = 0
    while i < len(a_list):
        if i < len(a_list) - 1 and a_list[i + 1] - a_list[i] < distance:
            average = (a_list[i] + a_list[i + 1]) // 2
            new_list.append(average)
            i += 2
        else:
            new_list.append(a_list[i])
            i += 1
    return new_list

"""
Es recomendable iterar varias veces sobre los mismos breakpoints
para conseguir un reemplazo estable delos bkps
"""
def merge_spurious_break_points_by_distance_until_stop(bkps, distance):
    #Merge breakpoints
    while True:
        new_bpks = merge_spurious_break_points_by_distance(bkps, distance)
        if new_bpks == bkps:
            break
        else:
            bkps = new_bpks
    return bkps

"""
Hay ventanas consecutivas cuyas medias no cambian considerablemente.
Las ventanas cuya media esta por debajo de umbral, se unen en una sola.
Unir es basicamente retirar el breakpoint que separa a las ventanas
consecutivas.
"""
def merge_breakpoints_by_window_mean(values, breakpoints, umbral):
    breakpoints = breakpoints.copy()
    if len(values) not in breakpoints:
        breakpoints.append(len(values))
    if len(breakpoints) != 1:
        last_break_point = 0
        new_break_points = []
        bkp_i = 0
        while bkp_i < len(breakpoints)-1:
            bkp_c = breakpoints[bkp_i]
            bkp_n = breakpoints[bkp_i+1]

            current_window = values[last_break_point:bkp_c]
            next_window = values[bkp_c:bkp_n]

            if np.abs(np.mean(current_window) - np.mean(next_window)) < umbral:
                new_break_points.append(bkp_n)
                bkp_i += 2
                last_break_point = bkp_n
            else:
                new_break_points.append(bkp_c)
                bkp_i += 1
                last_break_point = bkp_c

        breakpoints = new_break_points
    if len(values) in breakpoints:
        breakpoints.remove(len(values))
    return breakpoints

def merge_breakpoints_by_window_mean_until_stop(dataX, bkps, tresH):
    #Delete breakpoints
    while True:
        new_bpks = merge_breakpoints_by_window_mean(dataX,bkps,tresH)
        if new_bpks == bkps:
            break
        else:
            bkps = new_bpks
    return new_bpks
"""
Puede ocurrir que vengan breakpoints de dos fuentes distintas.
Por ejemplo, que venga los breakpoints del Alpha y del otro
lado breakpoints del coeficiente de difusion. Tomamos a esos
breakpoints (ya previamente tratados) y les hacemos
un ultimo refinamiento
"""
def merge_breakpoints_and_delete_spurious_of_different_data(A, B, distance):
    assert A[-1] == B[-1]
    length = A[-1]
    C = sorted(list(set(A+B)))
    bkps = merge_spurious_break_points_by_distance_until_stop(C,distance)
    if length not in bkps:
        bkps.append(length)
    return bkps

"""This section contains the 'core' loop of the stepfinder:
a single full iteration is done and a best fit is determined
@author: jkerssemakers march 2022       
"""

def break_point_detection_with_stepfinder(dataX, tresH=0.15, N_iter=100):
    demo = 0.0
    """This is the main, multi-pass loop of the autostepfinder
    @author: jkerssemakers march 2022"""
    FitX = 0 * dataX
    residuX = dataX - FitX
    newFitX, _, _, S_curve, best_shot = core.stepfindcore(
        residuX, demo, 0.15, N_iter
    )
    FitX = st.AppendFitX(newFitX, FitX, dataX)
    bkps = (np.where(np.diff(FitX.flatten())!=0)[0]+1).tolist()

    bkps = merge_spurious_break_points_by_distance_until_stop(bkps,4)
    bkps = merge_breakpoints_by_window_mean_until_stop(dataX,bkps,tresH)

    number_of_points = len(dataX)
    if number_of_points not in bkps:
        bkps.append(number_of_points)

    return bkps

def get_trajectories_from_2nd_andi_challenge_tiff_movie(
        tiff_movie,
        unet_network,
        expansion_factor=3,
        spt_max_distance_tolerance=15,
    ):
    tiff_movie = tiff_movie.copy()
    def get_id_of_position(a_dict, position):
        for an_id in a_dict:
            x_tuple = a_dict[an_id]['x']
            y_tuple = a_dict[an_id]['y']
            if x_tuple[0]<position[0]<x_tuple[1] and y_tuple[0]<position[1]<y_tuple[1]:
                return an_id
        return False

    #Get mask
    mask = tiff_movie[0]

    #Remove mask from the movie
    tiff_movie = tiff_movie[1:]

    # def conv(img, size):
    #     mask = np.ones([size, size], dtype = int) 
    #     mask = mask / (size*size)

    #     m, n = img.shape

    #     # Convolve the 3X3 mask over the image  
    #     img_new = np.zeros([m, n]) 
        
    #     for i in range(1, m-1): 
    #         for j in range(1, n-1): 
    #             temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2] 
                
    #             img_new[i, j]= temp 
                
    #     img_new = img_new.astype(np.uint8)
    #     return img_new
    # for frame_i in range(tiff_movie.shape[0]):
    #    tiff_movie[frame_i] = conv(tiff_movie[frame_i],3)
    #    tiff_movie[frame_i] = conv(tiff_movie[frame_i],5)

    #Get VIP ids
    vip_id_to_pixel_position = {}
    trajectory_vip_ids = np.unique(mask).tolist()
    trajectory_vip_ids.remove(255)

    #Extract size of "boxes" in the mask
    for an_id in trajectory_vip_ids:
        y_position, x_position = np.where(mask == an_id)

        vip_id_to_pixel_position[an_id] = {
            'x': (np.min(x_position)-expansion_factor, np.max(x_position)+expansion_factor),
            'y': (np.min(y_position)-expansion_factor, np.max(y_position)+expansion_factor),
        }

    #We get all the trajectories as dataframes
    dataframe = unet_network.predict(
        tiff_movie,
        pixel_size=1,
        extract_trajectories_as_dataframe=True,
        spt_max_distance_tolerance=spt_max_distance_tolerance,
        debug=False
    )

    #All trajectories are not VIP at the beginning
    dataframe['vip'] = False

    #vip_id_to_trajectories save all trajectories
    #within same 'boxes' in the mask
    vip_id_to_trajectories = defaultdict(lambda: [])
    first_frame_dataframe = dataframe[dataframe['frame']==0].copy()
    vip_trajectories_found = 0
    for track_id in first_frame_dataframe['traj_idx'].unique():
        track = first_frame_dataframe[first_frame_dataframe['traj_idx']==track_id]
        an_id = get_id_of_position(vip_id_to_pixel_position, track[['x','y']].values[0])
        if an_id is not False:
            vip_id_to_trajectories[an_id].append(track_id)

    for an_id in vip_id_to_trajectories:
        if len(vip_id_to_trajectories[an_id]) == 1:
            selected_track_id = vip_id_to_trajectories[an_id][0]
        else:
            box_width = (vip_id_to_pixel_position[an_id]['x'][1] - vip_id_to_pixel_position[an_id]['x'][0])/2
            box_height = (vip_id_to_pixel_position[an_id]['y'][1] - vip_id_to_pixel_position[an_id]['y'][0])/2

            box_center = (vip_id_to_pixel_position[an_id]['x'][0]+box_width,vip_id_to_pixel_position[an_id]['y'][0]+box_height)

            distances = []

            for trajectory_id in vip_id_to_trajectories[an_id]:
                x = first_frame_dataframe[first_frame_dataframe['traj_idx']==trajectory_id]['x'].values[0]
                y = first_frame_dataframe[first_frame_dataframe['traj_idx']==trajectory_id]['y'].values[0]

                distances.append(np.sqrt(((box_center[0]-x)**2)+((box_center[1]-y)**2)))

            selected_track_id = vip_id_to_trajectories[an_id][np.argmin(distances)]
        
        dataframe.loc[dataframe['traj_idx']==selected_track_id, 'vip'] = True
        vip_trajectories_found += 1

    assert len(trajectory_vip_ids)==vip_trajectories_found, f"{len(trajectory_vip_ids)}=={vip_trajectories_found}"
    return dataframe
