from TheoreticalModels import HopDiffusion, TrappingDiffusion
import numpy as np
import pandas as pd
import os

import ray


def batch_for_gen(generator, n=1):
    return_to_list = []
    
    for element in generator:
        if len(return_to_list) == n:
            yield return_to_list
            return_to_list = [element]
        else:
            return_to_list.append(element)
    
    yield return_to_list

ray.init()

@ray.remote
def simulate_trajectory(index, a_class):
    if os.path.exists(f'./single_simulations/single_data_{index}.csv'):
        return

    data = {
        'x': [],
        'y': [],
        't': [],
        'label': [],
    }

    trajectory = a_class.create_random_instance().simulate_trajectory(np.random.randint(100,1000), None)
    data['x'] += trajectory.get_noisy_x().tolist()
    data['y'] += trajectory.get_noisy_y().tolist()
    data['t'] += trajectory.get_time().tolist()
    data['label'] += [a_class.STRING_LABEL] * trajectory.length

    pd.DataFrame(data).to_csv(f'./single_simulations/single_data_{index}.csv', index=False)

import tqdm
for batch in tqdm.tqdm(batch_for_gen(range(70_000), n=1000)):
    ray.get([simulate_trajectory.remote(i, np.random.choice([HopDiffusion, TrappingDiffusion])) for i in batch])
