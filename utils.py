import numpy as np
from tifffile import TiffWriter, TiffFile

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
    C = sorted(list(set(A+B)))
    return merge_spurious_break_points_by_distance(C,distance)

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
