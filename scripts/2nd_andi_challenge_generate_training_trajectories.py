from DataSimulation import Andi2ndDataSimulation

for i in range(100):
    print(i)
    Andi2ndDataSimulation().simulate_phenomenological_trajectories(100_000,200,None,True,f'train_{i}', enable_parallelism=True)
