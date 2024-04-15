from TheoreticalModels import HopDiffusion, TrappingDiffusion
import numpy as np
import tqdm
import pandas as pd

data = {
    'x': [],
    'y': [],
    't': [],
    'label': [],
}

for i in tqdm.tqdm(list(range(150_000))):
    a_class = np.random.choice([HopDiffusion, TrappingDiffusion])
    trajectory = a_class.create_random_instance().simulate_trajectory(np.random.randint(100,1000), None)
    data['x'] += trajectory.get_noisy_x().tolist()
    data['y'] += trajectory.get_noisy_y().tolist()
    data['t'] += trajectory.get_time().tolist()
    data['label'] += [a_class.STRING_LABEL] * trajectory.length

pd.DataFrame(data).to_csv('hd_td_data.csv')
