from DataSimulation import Andi2ndDataSimulation

for i in range(100):
    Andi2ndDataSimulation().simulate_phenomenological_trajectories(100_000,100,None,True,f'train_{i}', enable_parallelism=True, ignore_boundary_effects=True)
