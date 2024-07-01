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

def delete_spurious_break_points(a_list, distancia=2):
    new_list = []
    i = 0
    while i < len(a_list):
        if i < len(a_list) - 1 and a_list[i + 1] - a_list[i] < distancia:
            average = (a_list[i] + a_list[i + 1]) // 2
            new_list.append(average)
            i += 2
        else:
            new_list.append(a_list[i])
            i += 1
    return new_list

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
        residuX, demo, tresH, N_iter
    )
    FitX = st.AppendFitX(newFitX, FitX, dataX)
    bkps = (np.where(np.diff(FitX.flatten())!=0)[0]+1).tolist()

    while True:
        new_bpks = delete_spurious_break_points(bkps, distancia=4)

        if new_bpks == bkps:
            break
        else:
            bkps = new_bpks
    return bkps
