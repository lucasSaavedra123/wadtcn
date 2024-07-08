import numpy as np
import matplotlib.pyplot as plt

from CONSTANTS import VALIDATION_SET_SIZE_PER_EPOCH, ALPHA_ACCEPTANCE_THRESHOLD, D_ACCEPTANCE_THRESHOLD
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter
from utils import break_point_detection_with_stepfinder, merge_breakpoints_and_delete_spurious_of_different_data


trajectories = Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_regression_training(VALIDATION_SET_SIZE_PER_EPOCH,200,None,True,'val', ignore_boundary_effects=True)
limit = 100
sigma = 0.12
for t in trajectories:
    t.x = (np.array(t.x) + np.random.randn(t.length)*sigma).tolist()
    t.y = (np.array(t.y) + np.random.randn(t.length)*sigma).tolist()

alpha_network = WavenetTCNSingleLevelAlphaPredicter(200, None, simulator=Andi2ndDataSimulation)
alpha_network.load_as_file()

diffusion_coefficient_network = WavenetTCNSingleLevelDiffusionCoefficientPredicter(200, None, simulator=Andi2ndDataSimulation)
diffusion_coefficient_network.load_as_file()

np.random.shuffle(trajectories)
trajectories = trajectories[:limit]

for trajectory in trajectories:
    alpha_result = alpha_network.predict([trajectory])[0,:,0]*2
    d_result = diffusion_coefficient_network.predict([trajectory])[0,:,0]

    fig, ax = plt.subplots(2,1)

    alpha_breakpoints = break_point_detection_with_stepfinder(alpha_result, 3, tresH=ALPHA_ACCEPTANCE_THRESHOLD)
    d_breakpoints = break_point_detection_with_stepfinder(d_result, 3, tresH=D_ACCEPTANCE_THRESHOLD)

    ax[0].scatter(range(trajectory.length), alpha_result)
    for bkp in alpha_breakpoints:
        ax[0].axvline(bkp, color='blue')
    ax[0].plot(trajectory.info['alpha_t'])
    ax[0].set_title('Alpha')
    ax[0].set_ylim([0,2])

    ax[1].scatter(range(trajectory.length), d_result)
    for bkp in d_breakpoints:
        ax[1].axvline(bkp, color='blue')
    ax[1].plot(np.log10(trajectory.info['d_t']))
    ax[1].set_title('Diffusion Coefficient')
    #ax[1].set_ylim([-12,6])

    #Show final breakpoints
    final_breakpoints = merge_breakpoints_and_delete_spurious_of_different_data(alpha_breakpoints, d_breakpoints, 4)

    for bkp in final_breakpoints:
        ax[0].axvline(bkp, color='red', linewidth=2)
        ax[1].axvline(bkp, color='red', linewidth=2)

    plt.show()
