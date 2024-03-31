import numpy as np
from tifffile import TiffWriter, TiffFile


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
