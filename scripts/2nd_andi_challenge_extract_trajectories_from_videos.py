from os.path import join
import tqdm


from PredictiveModel.UNetSingleParticleTracker import UNetSingleParticleTracker
from utils import tiff_movie_path_to_numpy_array, get_trajectories_from_2nd_andi_challenge_tiff_movie

PUBLIC_DATA_PATH = './public_data_challenge_v0'
PATH_TRACK_1 = './track_1'

N_EXP = 10
N_FOVS = 30

unet_network = UNetSingleParticleTracker(128,128,2)
unet_network.load_as_file()

for exp in tqdm.tqdm(list(range(N_EXP))):
    for fov in range(N_FOVS):
        print(exp,fov)
        tiff_file_path = join(PUBLIC_DATA_PATH, PATH_TRACK_1, f'exp_{exp}', f'videos_fov_{fov}.tiff')
        tiff_movie = tiff_movie_path_to_numpy_array(tiff_file_path)
        #if exp==4:
        #    dataframe = get_trajectories_from_2nd_andi_challenge_tiff_movie(tiff_movie, unet_network, spt_max_distance_tolerance=7)
        #else:
        dataframe = get_trajectories_from_2nd_andi_challenge_tiff_movie(tiff_movie, unet_network)
        dataframe.to_csv(join(PUBLIC_DATA_PATH, PATH_TRACK_1, f'exp_{exp}', f'trajs_fov_{fov}.csv'), index=False)
