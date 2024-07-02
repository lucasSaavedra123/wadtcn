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

def merge_spurious_break_points(a_list, distance):
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

def merge_breakpoints(A, B, distance):
    i, j = 0, 0
    C = []
    len_A, len_B = len(A), len(B)
    
    while i < len_A and j < len_B:
        if abs(A[i] - B[j]) <= distance:
            C.append((A[i] + B[j]) // 2)
            i += 1
            j += 1
        elif A[i] < B[j]:
            C.append(A[i])
            i += 1
        else:
            C.append(B[j])
            j += 1
    
    while i < len_A:
        C.append(A[i])
        i += 1
    
    while j < len_B:
        C.append(B[j])
        j += 1
    
    return merge_spurious_break_points(C,distance)

def merge_windows_defined_by_break_points(values, breakpoints, umbral):
    if len(breakpoints) == 1:
        return breakpoints
    last_break_point = 0

    new_break_points = []

    for bkp_i in range(0,len(breakpoints)-1):
        bkp_c = breakpoints[bkp_i]
        bkp_n = breakpoints[bkp_i+1]

        current_window = values[last_break_point:bkp_c]
        next_window = values[bkp_c:bkp_n]

        if np.abs(np.mean(current_window) - np.mean(next_window)) < umbral:
            new_break_points.append(bkp_n)
        else:
            new_break_points.append(bkp_c)
        last_break_point = bkp_c
    if len(values) not in new_break_points:
        new_break_points.append(len(values))
    return new_break_points

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

    #Merge breakpoints
    while True:
        new_bpks = merge_spurious_break_points(bkps, 4)
        if new_bpks == bkps:
            break
        else:
            bkps = new_bpks

    number_of_points = len(dataX)
    if number_of_points not in bkps:
        bkps.append(number_of_points)

    #Delete breakpoints
    while True:
        new_bpks = merge_windows_defined_by_break_points(dataX,bkps,tresH)
        if new_bpks == bkps:
            break
        else:
            bkps = new_bpks

    return bkps
