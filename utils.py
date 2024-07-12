import numpy as np
from tifffile import TiffWriter, TiffFile
from collections import defaultdict
from statistics import mode

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
        if i < len(a_list) - 1 and a_list[i + 1] - a_list[i] <= distance:
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
def merge_breakpoints_by_window_criterion(values, breakpoints, umbral=0.5, criterion='mean'):
    class Window:
        def __init__(self, values, initial_index, final_index):
            self.values = np.array(values).tolist()
            self.indexes = [initial_index, final_index]

        @classmethod
        def merge_windows(cls, window_one, window_two):
            return Window(
                window_one.values + window_two.values,
                window_one.indexes[0],
                window_two.indexes[1]
            )

        def representative_value(self):
            if criterion == 'mean':
                return np.mean(self.values)
            else:
                return mode(self.values)
        
        def statically_overlap_with(self, other_window):
            self_range = [
                np.mean(self.values) - np.std(self.values),
                np.mean(self.values) + np.std(self.values)
            ]
            other_window_range = [
                np.mean(other_window.values) - np.std(other_window.values),
                np.mean(other_window.values) + np.std(other_window.values)
            ]

            if other_window_range[0] < self_range[0]:
                self_range, other_window_range = other_window_range, self_range

            #overlap = self_range[0] <= other_window_range[1] and self_range[1] >= other_window_range[0]
            overlap = max(0, min(self_range[1], other_window_range[1]) - max(self_range[0], other_window_range[0]))
            return overlap != 0

    breakpoints = breakpoints.copy()
    if len(values) not in breakpoints:
        breakpoints.append(len(values))
    initial_index = 0
    windows = []

    for bkp in breakpoints:
        windows.append(
            Window(
                values[initial_index:bkp],
                initial_index,
                bkp
            )
        )
        initial_index = bkp

    window_index = 0

    while window_index < len(windows) - 1:
        if criterion == 'mean':
            condition = windows[window_index].statically_overlap_with(windows[window_index+1])#abs(windows[window_index].representative_value() - windows[window_index+1].representative_value()) < umbral
        else:
            condition = windows[window_index].representative_value() == windows[window_index+1].representative_value()
        if condition:
            new_window = Window.merge_windows(
                windows[window_index],
                windows[window_index+1]
            )
            windows.remove(windows[window_index])
            windows[window_index] = new_window
        window_index += 1

    breakpoints = [w.indexes[1] for w in windows]

    if len(values) in breakpoints:
        breakpoints.remove(len(values))
    return breakpoints

def merge_breakpoints_by_window_criterion_until_stop(dataX, bkps, criterion='mean'):
    #Delete breakpoints
    while True:
        new_bpks = merge_breakpoints_by_window_criterion(dataX,bkps, criterion)
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
def merge_breakpoints_and_delete_spurious_of_different_data(A, B, distance, EXTRA=[]):
    if len(EXTRA) != 0:
        assert A[-1] == B[-1] == EXTRA[-1]
        C = sorted(list(set(A+B+EXTRA)))
    else:
        assert A[-1] == B[-1]
        C = sorted(list(set(A+B)))
    length = A[-1]
    bkps = merge_spurious_break_points_by_distance_until_stop(C,distance)
    if length not in bkps:
        bkps.append(length)
    if length - 1 in bkps:
        bkps.remove(length - 1)
    return bkps

"""This section contains the 'core' loop of the stepfinder:
a single full iteration is done and a best fit is determined
@author: jkerssemakers march 2022       
"""

def break_point_detection_with_stepfinder(dataX, distance, N_iter=100):
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

    bkps = merge_spurious_break_points_by_distance_until_stop(bkps,distance)
    bkps = merge_breakpoints_by_window_criterion_until_stop(dataX,bkps)

    number_of_points = len(dataX)
    if number_of_points not in bkps:
        bkps.append(number_of_points)
    if len(bkps) > 1 and bkps[-1] - bkps[-2] <= distance:
        bkps.remove(bkps[-2])
    return bkps

"""
Hay arrays de estados. Es decir, arrays discretos.
Estos no pasan por stepfinder. Sino que los breakpoints
son directamente los cambias de transición que detectó
la red y retira ventanas por moda, no por promedio.
"""

def break_point_discrete_detection(dataX, distance):
    number_of_points = len(dataX)
    bkps = (np.where(np.diff(dataX)!=0)[0]+1).tolist()
    if number_of_points in bkps:
        bkps.remove(number_of_points)
    bkps = merge_spurious_break_points_by_distance_until_stop(bkps,4)
    bkps = merge_breakpoints_by_window_criterion_until_stop(dataX,bkps,criterion='mode')

    if number_of_points not in bkps:
        bkps.append(number_of_points)
    if len(bkps) > 1 and bkps[-1] - bkps[-2] <= distance:
        bkps.remove(bkps[-2])
    return bkps

"""
Dado los checkpoints, voy a querer actualizar los valores de
cada ventana por la media
"""
def refine_values_and_states_following_breakpoints(alpha, diffusion_coefficient, state, cp):
    assert len(alpha) == cp[-1]
    assert len(diffusion_coefficient) == cp[-1]
    assert len(state) == cp[-1]
    alpha = alpha.copy()
    diffusion_coefficient = diffusion_coefficient.copy()
    state = state.copy()
    last_break_point = 0

    for cp_i in range(len(cp)):
        cp_initial = last_break_point
        cp_final = cp[cp_i]

        alpha[cp_initial:cp_final] = np.mean(alpha[cp_initial:cp_final])
        diffusion_coefficient[cp_initial:cp_final] = np.mean(diffusion_coefficient[cp_initial:cp_final])
        state[cp_initial:cp_final] = mode(state[cp_initial:cp_final])
        last_break_point = cp_final

    return alpha, diffusion_coefficient, state

def get_trajectories_from_2nd_andi_challenge_tiff_movie(
        tiff_movie,
        unet_network,
        expansion_factor=3,
        spt_max_distance_tolerance=15,
        spt_adaptive_stop=3,
        assertion = True
    ):
    tiff_movie = tiff_movie.copy()
    def get_vip_ids_of_position(a_dict, position):
        ids = []
        for vip_id in a_dict:
            x_tuple = a_dict[vip_id]['x']
            y_tuple = a_dict[vip_id]['y']
            if x_tuple[0]<position[0]<x_tuple[1] and y_tuple[0]<position[1]<y_tuple[1]:
                ids.append(vip_id)
        return False if len(ids)==0 else ids

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
    for vip_id in trajectory_vip_ids:
        y_position, x_position = np.where(mask == vip_id)

        vip_id_to_pixel_position[vip_id] = {
            'x': (np.min(x_position)-expansion_factor, np.max(x_position)+expansion_factor),
            'y': (np.min(y_position)-expansion_factor, np.max(y_position)+expansion_factor),
        }

    #We get all the trajectories as dataframes
    dataframe = unet_network.predict(
        tiff_movie,
        pixel_size=1,
        extract_trajectories_as_dataframe=True,
        spt_max_distance_tolerance=spt_max_distance_tolerance,
        spt_adaptive_stop=spt_adaptive_stop,
        debug=False,
        intensity_filter=False
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
        vip_ids = get_vip_ids_of_position(vip_id_to_pixel_position, track[['x','y']].values[0])
        if vip_ids is not False:
            for vip_id in vip_ids:
                vip_id_to_trajectories[vip_id].append(track_id)

    dataframe['new_traj_idx'] = -1
    for vip_id in vip_id_to_trajectories:
        if len(vip_id_to_trajectories[vip_id]) == 1:
            selected_track_id = vip_id_to_trajectories[vip_id][0]
        else:
            box_width = (vip_id_to_pixel_position[vip_id]['x'][1] - vip_id_to_pixel_position[vip_id]['x'][0])/2
            box_height = (vip_id_to_pixel_position[vip_id]['y'][1] - vip_id_to_pixel_position[vip_id]['y'][0])/2

            box_center = (vip_id_to_pixel_position[vip_id]['x'][0]+box_width,vip_id_to_pixel_position[vip_id]['y'][0]+box_height)

            distances = []

            for trajectory_id in vip_id_to_trajectories[vip_id]:
                x = first_frame_dataframe[first_frame_dataframe['traj_idx']==trajectory_id]['x'].values[0]
                y = first_frame_dataframe[first_frame_dataframe['traj_idx']==trajectory_id]['y'].values[0]

                distances.append(np.sqrt(((box_center[0]-x)**2)+((box_center[1]-y)**2)))

            selected_track_id = vip_id_to_trajectories[vip_id][np.argmin(distances)]

            for aux_id in vip_id_to_trajectories:
                if aux_id != vip_id and selected_track_id in vip_id_to_trajectories[aux_id]:
                    vip_id_to_trajectories[aux_id].remove(selected_track_id)

        dataframe.loc[dataframe['traj_idx']==selected_track_id, 'vip'] = True
        dataframe.loc[dataframe['traj_idx']==selected_track_id, 'new_traj_idx'] = vip_id
        vip_trajectories_found += 1

    def give_id():
        an_id = 0
        while True:
            if an_id not in trajectory_vip_ids:
                yield an_id
            an_id += 1

    id_iterator = give_id()

    for index, row in dataframe.iterrows():
        row.traj_idx = next(id_iterator) if not row.vip else row.new_traj_idx

    dataframe = dataframe.drop(['new_traj_idx'], axis=1)
    assert not assertion or len(trajectory_vip_ids)==vip_trajectories_found, f"{len(trajectory_vip_ids)}=={vip_trajectories_found}"
    return dataframe

"""
Below code was extracted from
https://colab.research.google.com/drive/1Jir3HxTZ-au8L56ZrNHGxfBD0XlDkOMl?usp=sharing#scrollTo=BoNTgNwNMVWW
"""
def fit_position_within_image(ROI):
  #assert ROI.shape[0] % 2 == 1 and ROI.shape[1] % 2 == 1
  ROIradius = ROI.shape[0]//2
  #Perform 2D Fourier transform over the complete ROI
  ROI_F = np.fft.fft2(ROI)

  #We have to calculate the phase angle of array entries [0,1] and [1,0] for
  #the sub-pixel x and y values, respectively
  #This phase angle can be calculated as follows:
  xangle = np.arctan(ROI_F[0,1].imag/ROI_F[0,1].real) - np.pi
  #Correct in case it's positive
  if xangle > 0:
    xangle -= 2*np.pi
  #Calculate position based on the ROI radius
  PositionX = abs(xangle)/(2*np.pi/(ROIradius*2+1))+0.5

  #Do the same for the Y angle and position
  yangle = np.arctan(ROI_F[1,0].imag/ROI_F[1,0].real) - np.pi
  if yangle > 0:
    yangle -= 2*np.pi
  PositionY = abs(yangle)/(2*np.pi/(ROIradius*2+1))+0.5

  return [PositionX, PositionY]
