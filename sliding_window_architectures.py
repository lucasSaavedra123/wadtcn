import numpy as np
import matplotlib.pyplot as plt

#from PredictiveModel.SlidingWindowHurstExponentPredicter import SlidingWindowHurstExponentPredicter
from PredictiveModel.WavenetTCNSlidingWindowfBM import WavenetTCNSlidingWindowfBM
from DataSimulation import CustomDataSimulation
from TheoreticalModels.BrownianMotion import BrownianMotion
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory

DatabaseHandler.connect_over_network(None, None, '192.168.0.101', 'MINFLUX_DATA')

LOAD_BOOLEAN = True

diffusion_coefficient_sliding_window = WavenetTCNSlidingWindowfBM(25,None, simulator=CustomDataSimulation)
#diffusion_coefficient_sliding_window.enable_early_stopping()
#diffusion_coefficient_sliding_window.fit()
#diffusion_coefficient_sliding_window.save_as_file()
diffusion_coefficient_sliding_window.load_as_file()

#diffusion_coefficient_sliding_window.plot_predicted_and_ground_truth_histogram()

for ts in Trajectory.objects():
    #a = diffusion_coefficient_sliding_window.simulate_trajectories(1,True,False)[0]
    #b = diffusion_coefficient_sliding_window.simulate_trajectories(1,True,False)[0]

    #ts = [a.merge_trajectories(b)]
    value = diffusion_coefficient_sliding_window.predict([ts])



    plt.plot(value[0])
    plt.yscale('log')
    plt.show()

DatabaseHandler.disconnect()

#diffusion_coefficient_sliding_window.plot_predicted_and_ground_truth_histogram()








"""
if LOAD_BOOLEAN:
    hurst_exponent_sliding_window.load_as_file()
else:
    hurst_exponent_sliding_window.enable_early_stopping()
    hurst_exponent_sliding_window.fit()
    hurst_exponent_sliding_window.save_as_file()

#hurst_exponent_sliding_window.plot_predicted_and_ground_truth_distribution()
#hurst_exponent_sliding_window.plot_bias()
hurst_exponent_sliding_window.plot_predicted_and_ground_truth_histogram()

if LOAD_BOOLEAN:
    diffusion_coefficient_sliding_window.load_as_file()
else:
    diffusion_coefficient_sliding_window.enable_early_stopping()
    diffusion_coefficient_sliding_window.fit()
    diffusion_coefficient_sliding_window.save_as_file()

#diffusion_coefficient_sliding_window.plot_predicted_and_ground_truth_distribution()
#diffusion_coefficient_sliding_window.plot_bias()
diffusion_coefficient_sliding_window.plot_predicted_and_ground_truth_histogram()
"""