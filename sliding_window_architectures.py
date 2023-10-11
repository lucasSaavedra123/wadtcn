import numpy as np
import matplotlib.pyplot as plt

from PredictiveModel.SlidingWindowHurstExponentPredicter import SlidingWindowHurstExponentPredicter
from PredictiveModel.SlidingWindowDiffusionCoefficientPredicter import SlidingWindowDiffusionCoefficientPredicter
from DataSimulation import CustomDataSimulation

LOAD_BOOLEAN = True

hurst_exponent_sliding_window = SlidingWindowHurstExponentPredicter(25,25*0.001, model='fbm', simulator=CustomDataSimulation)
diffusion_coefficient_sliding_window = SlidingWindowDiffusionCoefficientPredicter(25,25*0.001, model='fbm', simulator=CustomDataSimulation)

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
