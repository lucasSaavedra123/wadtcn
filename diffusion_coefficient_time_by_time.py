#import numpy as np
#import matplotlib.pyplot as plt

#from PredictiveModel.SlidingWindowHurstExponentPredicter import SlidingWindowHurstExponentPredicter
#from PredictiveModel.WavenetTCNSlidingWindowfBM import WavenetTCNSlidingWindowfBM
#from DataSimulation import CustomDataSimulation
#from TheoreticalModels.BrownianMotion import BrownianMotion
#from DatabaseHandler import DatabaseHandler
#from Trajectory import Trajectory

#DatabaseHandler.connect_over_network(None, None, '192.168.0.101', 'MINFLUX_DATA')
#DatabaseHandler.disconnect()

from step.data import *
from step.utils import *
from step.models import *
from fastai.vision.all import *

class LocalizationNoise(ItemTransform):
    "Add localization noise to the trajectories."
    def __init__(self, noise_lvls): self.noise_lvls = tensor(noise_lvls)
    def encodes(self, sample):
        x, y = sample
        idx = torch.randint(self.noise_lvls.shape[0], (1,))
        noise = self.noise_lvls[idx]
        noisy_x = x + 10**(noise)*torch.randn_like(x)
        return noisy_x - noisy_x[0], y

n_per_set = 12000
max_t = 200
dim = 2
Ds = np.logspace(-3, 3, 1000) 
cps = [1, 2, 3, 4]
ds_fun = partial(create_bm_segmentation_dataset,
                 max_t=max_t, dim=dim, Ds=Ds, save=False)

datasets = [ds_fun(n_per_set, n_change_points=n_cp) for n_cp in cps]
dataset = combine_datasets(datasets)

dim = 2
dls = get_segmentation_dls(dim=dim, n_change='1_to_4', name='train',
                           tfm_y=torch.log10, bm=True)
dls.device = default_device()

model = XResAttn(2, n_class=1, stem_szs=(64,), conv_blocks=[1, 1, 1],
                 block_szs=[128, 256, 512], pos_enc=False, 
                 n_encoder_layers=4, dim_ff=512, nhead_enc=8,
                 linear_layers=[], norm=False, yrange=(-3.1, 3.1))

learn = Learner(dls, model, loss_func=L1LossFlat(), model_dir=MODEL_PATH)



model.to(default_device())