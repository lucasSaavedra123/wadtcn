import numpy as np
import matplotlib.pyplot as plt

from CONSTANTS import VALIDATION_SET_SIZE_PER_EPOCH, ALPHA_ACCEPTANCE_THRESHOLD, D_ACCEPTANCE_THRESHOLD
from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNSingleLevelDiffusionCoefficientPredicter import WavenetTCNSingleLevelDiffusionCoefficientPredicter
from utils import break_point_detection_with_stepfinder, merge_breakpoints_and_delete_spurious_of_different_data, break_point_discrete_detection, refine_values_and_states_following_breakpoints
from andi_datasets.utils_challenge import label_continuous_to_list, single_changepoint_error


classification_network = WavenetTCNMultiTaskClassifierSingleLevelPredicter(200, None, simulator=Andi2ndDataSimulation)
classification_network.load_as_file()
alpha_network = WavenetTCNSingleLevelAlphaPredicter(200, None, simulator=Andi2ndDataSimulation)
alpha_network.load_as_file()
diffusion_coefficient_network = WavenetTCNSingleLevelDiffusionCoefficientPredicter(200, None, simulator=Andi2ndDataSimulation)
diffusion_coefficient_network.load_as_file()

while True:
    retry = True
    while retry:
        try:
            trajectories = []
            while len(trajectories) == 0:
                trajectories = Andi2ndDataSimulation().simulate_challenge_trajectories(filter=True)
            retry = False
        except:
            retry = True
    for t in trajectories:
        plt.plot(t.get_noisy_x(), t.get_noisy_y())
    plt.show()
    retry = True

    while retry:
        try:
            value = int(input('0 (Skip) or 1 (Continue): '))
            retry = False
        except ValueError:
            pass
    
    if value==0:
        continue

    for trajectory in trajectories:
        alpha_result = alpha_network.predict([trajectory])[0,:,0]*2
        d_result = diffusion_coefficient_network.predict([trajectory])[0,:,0]
        state_result = np.argmax(classification_network.predict([trajectory])[0], axis=1)

        fig, ax = plt.subplots(3,1)

        alpha_breakpoints = break_point_detection_with_stepfinder(alpha_result, 3, 0.1)
        d_breakpoints = break_point_detection_with_stepfinder(d_result, 3, 0.1)
        state_breakpoints = break_point_discrete_detection(state_result,3)

        final_breakpoints = merge_breakpoints_and_delete_spurious_of_different_data(
            alpha_breakpoints,
            d_breakpoints,
            3,
            EXTRA=state_breakpoints
        )

        refined_alpha, refined_d, refined_state = refine_values_and_states_following_breakpoints(alpha_result, d_result, state_result, final_breakpoints)

        ax[0].plot(trajectory.info['alpha_t'])
        ax[0].plot(refined_alpha, color='black')
        ax[0].scatter(range(trajectory.length), alpha_result)
        ax[0].set_title('Alpha')

        if d_result.max() - d_result.min() < 2:
            ax[1].plot(trajectory.info['d_t'])
            ax[1].plot(10**refined_d, color='black')
            ax[1].scatter(range(trajectory.length), 10**d_result)
        else:
            ax[1].plot(np.log10(trajectory.info['d_t']))
            ax[1].plot(refined_d, color='black')
            ax[1].scatter(range(trajectory.length), d_result)

        ax[1].set_title('Diffusion Coefficient')

        ax[2].plot(trajectory.info['state_t'])
        ax[2].plot(refined_state, color='black')
        ax[2].scatter(range(trajectory.length), state_result)
        ax[2].set_title('Single-level classification')

        for bkp in alpha_breakpoints:
            ax[0].axvline(bkp, color='blue')
        for bkp in d_breakpoints:
            ax[1].axvline(bkp, color='blue')
        for bkp in state_breakpoints:
            ax[2].axvline(bkp, color='blue')
        for bkp in final_breakpoints:
            ax[0].axvline(bkp, color='red', linewidth=2)
            ax[1].axvline(bkp, color='red', linewidth=2)
            ax[2].axvline(bkp, color='red', linewidth=2)

        gt_labs = np.zeros((trajectory.length,3))
        gt_labs[:,0] = trajectory.info['alpha_t']
        gt_labs[:,1] = trajectory.info['d_t']
        gt_labs[:,2] = trajectory.info['state_t']

        gt_cp = label_continuous_to_list(gt_labs)[0]
        rmse, jcs = single_changepoint_error(gt_cp, np.array(final_breakpoints), threshold = 10)

        fig.suptitle(f'RMSE={np.round(rmse,2)}, JCS={np.round(jcs,2)}')

        plt.show()
        """
        trajectory.plot_andi_2()

        del trajectory.info['d_t']
        del trajectory.info['alpha_t']
        del trajectory.info['state_t']

        trajectory.info['d_t'] = 10**np.array(d_result)
        trajectory.info['alpha_t'] = 2*np.array(alpha_result)
        trajectory.info['state_t'] = state_result
        trajectory.plot_andi_2()
        """
