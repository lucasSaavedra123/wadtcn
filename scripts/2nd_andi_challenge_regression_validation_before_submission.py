import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

from CONSTANTS import VALIDATION_SET_SIZE_PER_EPOCH, ALPHA_ACCEPTANCE_THRESHOLD, D_ACCEPTANCE_THRESHOLD
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter
from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter
from utils import break_point_detection_with_stepfinder, merge_breakpoints_and_delete_spurious_of_different_data, refine_values_and_states_following_breakpoints, merge_spurious_break_points_by_distance_until_stop
from andi_datasets.utils_challenge import label_continuous_to_list, single_changepoint_error

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
"""
cp_network = WavenetTCNSingleLevelChangePointPredicter(200, None, simulator=Andi2ndDataSimulation)
cp_network.load_as_file()
"""

np.random.shuffle(trajectories)
trajectories = trajectories[:limit]

for trajectory in trajectories:
    alpha_result = alpha_network.predict([trajectory])[0,:,0]*2
    d_result = diffusion_coefficient_network.predict([trajectory])[0,:,0]

    alpha_breakpoints = break_point_detection_with_stepfinder(alpha_result, 3, 0.1)
    d_breakpoints = break_point_detection_with_stepfinder(d_result, 3, 0.1)
    fig, ax = plt.subplots(2,1)
    ax[0].set_title('Alpha')
    ax[1].set_title('Diffusion Coefficient')

    #Show final breakpoints
    final_breakpoints = merge_breakpoints_and_delete_spurious_of_different_data(alpha_breakpoints, d_breakpoints, 3)
    """
    cp_network_breakpoints = np.where(cp_network.predict([trajectory])[0,:,0] == 1)[0].tolist()
    cp_network_breakpoints = merge_spurious_break_points_by_distance_until_stop(cp_network_breakpoints, 3)
    if trajectory.length not in cp_network_breakpoints:
        cp_network_breakpoints.append(trajectory.length)
    if 0 in cp_network_breakpoints:
        cp_network_breakpoints.remove(0)

    final_breakpoints = merge_breakpoints_and_delete_spurious_of_different_data(cp_network_breakpoints, final_breakpoints, 3)
    """
    refined_alpha, refined_d, _ = refine_values_and_states_following_breakpoints(alpha_result, d_result, 2*np.ones(200), final_breakpoints)

    ax[0].scatter(range(trajectory.length), alpha_result, color='green')
    ax[0].plot(trajectory.info['alpha_t'])
    ax[0].plot(refined_alpha, color='black')
    ax[0].set_ylim([0,2])

    if d_result.max() - d_result.min() < 2:
        ax[1].plot(trajectory.info['d_t'])
        ax[1].plot(10**refined_d, color='black')
        ax[1].scatter(range(trajectory.length), 10**d_result, color='green')
        ax[1].set_ylabel('Absolute Diffusion Coefficient')
    else:
        ax[1].plot(np.log10(trajectory.info['d_t']))
        ax[1].plot(refined_d, color='black')
        ax[1].scatter(range(trajectory.length), d_result, color='green')
        ax[1].set_ylabel('Logarithmic Diffusion Coefficient')

    for bkp in alpha_breakpoints:
        ax[0].axvline(bkp, color='blue', linewidth=3)
    for bkp in d_breakpoints:
        ax[1].axvline(bkp, color='blue', linewidth=3)

    gt_labs = np.zeros((200,3))
    gt_labs[:,0] = trajectory.info['alpha_t']
    gt_labs[:,1] = trajectory.info['d_t']
    gt_labs[:,2] = trajectory.info['state_t']

    gt_cp = label_continuous_to_list(gt_labs)[0]
    rmse, jcs = single_changepoint_error(gt_cp, np.array(final_breakpoints), threshold = 10)

    fig.suptitle(f'RMSE={np.round(rmse,2)}, JCS={np.round(jcs,2)}')

    for bkp in final_breakpoints:
        ax[0].axvline(bkp, color='red', linewidth=1)
        ax[1].axvline(bkp, color='red', linewidth=1)

    plt.show()
