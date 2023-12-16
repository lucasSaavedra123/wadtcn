import numpy as np
import matplotlib.pyplot as plt

#from PredictiveModel.SlidingWindowHurstExponentPredicter import SlidingWindowHurstExponentPredicter
from PredictiveModel.WavenetTCNSlidingWindowfBM import WavenetTCNSlidingWindowfBM
from DataSimulation import CustomDataSimulation
from TheoreticalModels.BrownianMotion import BrownianMotion

LOAD_BOOLEAN = True

diffusion_coefficient_sliding_window = WavenetTCNSlidingWindowfBM(25,25*0.01, simulator=CustomDataSimulation)
diffusion_coefficient_sliding_window.enable_early_stopping()
diffusion_coefficient_sliding_window.fit()
diffusion_coefficient_sliding_window.save_as_file()
#diffusion_coefficient_sliding_window.load_as_file()

diffusion_coefficient_sliding_window.plot_predicted_and_ground_truth_histogram()

while True:
    new_d = np.random.uniform(10**-3,10**0)
    #new_trajectory = FractionalBrownianMotion(np.random.uniform(0,1), 10**np.random.choice(simulated_Ds)).simulate_trajectory(self.trajectory_length, self.trajectory_time, from_andi=False)
    new_trajectory = BrownianMotion(new_d).simulate_trajectory(250, 250 * 0.01, from_andi=False) 
    ts = [new_trajectory]
    value = diffusion_coefficient_sliding_window.predict(ts)
    print(new_d)
    plt.plot(value[0])
    plt.ylim([10**-3, 10**0])
    plt.show()



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