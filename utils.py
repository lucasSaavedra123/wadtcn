import numpy as np
from tifffile import TiffWriter, TiffFile
from sklearn.neighbors import NearestNeighbors


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

#Code From https://drive.google.com/drive/u/0/folders/1lOKvC_L2fb78--uwz3on4lBzDGVum8Mc
def create_trajectories(tracked_data,frame1,frame2,maxDistance,tracksCounter):
  #Make sub-matrix of this frame and the next frame
  framematrix = tracked_data[tracked_data[:,0]==frame1,:]
  nextframematrix = tracked_data[tracked_data[:,0]==frame2,:]

  #Now we check that there are localizations in both of these frames - if one
  #of the frames does not have localizations, we cannot try to track the
  #localizations. Note that we already know that in this case, frame 0 and
  #frame 1 have localizations, but this will not always be the case.
  if (len(framematrix) > 0 and len(nextframematrix) > 0):
    #Now we can find all the nearest neighbours between all localizations
    # on this frame and on the next frame
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(nextframematrix[:,1:3])
    NearestNeighbors(n_neighbors=1)
    foundnn = neigh.kneighbors(framematrix[:,1:3])
    foundnn = np.asarray(foundnn)

    # Now we have to select only those entries that are < maxDistance. We want
    # to extract the IDs of the localizations in frame n+1 that belong to them.
    NeighbourIDs = foundnn[:,foundnn[0] < maxDistance][1].astype(int)
    # We also want to extract the ID of the original localizations by
    # finding the row index of the nearest neighbours.
    OriginIDs = np.where(foundnn[0] < maxDistance)[0].astype(int)

    # For every found neighbour, we make both neighbours the same track-id.
    # First, we check that the neighbour on frame n has or doesn't have an
    # track-id yet, then we set the neighbour on frame n+1
    # to the same value, or to a new track-id if none is assigned yet
    # We loop over all found IDs
    for i in range(0,len(NeighbourIDs)):
      #We get the localization-ID of the neighbour in frame n+1
      neighbourID = NeighbourIDs[i]
      #We also get the localization-ID of the original localization in frame n
      originID = OriginIDs[i]
      #Prevent linkage if the neighbour in frame n+1 is already linked -
      #this will not happen in this example, but it might happen when
      #skipping frames in later modules.
      if nextframematrix[neighbourID,4] == 0:
        # We check that the localization that will be included in a trajectory
        # is not yet part of an existing trajectory - if it is part of
        # an existing track, the index in column 4 will be higher than 0
        if framematrix[originID,4] == 0:
          #If it's not linked yet, set it and the neighbour to a new track-id value
          framematrix[originID,4] = tracksCounter
          nextframematrix[neighbourID,4] = tracksCounter
          tracksCounter += 1
        else:
          #If it is linked, set it to the track-id value of the neighbour in frame n
          nextframematrix[neighbourID,4] = framematrix[originID,4]

        #Finally, we  provide the distance to the next emitter in the
        #track on the next frame
        framematrix[originID][5] = foundnn[0][originID][0]

  #Now we have fully filled frame matrix and nextframematrix variables, but
  #we need to fill these back in into the original tracked_data matrix. We do
  #this by looking up the values via the idCounter value
  if len(framematrix) > 0:
      tracked_data[tracked_data[:,0]==frame1] = framematrix
  if len(nextframematrix) > 0:
      tracked_data[tracked_data[:,0]==frame2] = nextframematrix
      
  #Return the required parameters
  return tracked_data, tracksCounter
