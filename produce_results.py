import tqdm
import matplotlib.pyplot as plt
import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation
from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME, IMMOBILE_THRESHOLD
from TheoreticalModels import ALL_MODELS, SBM_MODELS, FBM_MODELS
from Trajectory import Trajectory
from collections import Counter, defaultdict


"""
def get_classification_error(steps_range, exp_label, exp_cond, net_name):
    if net_name == 'L1 Network':
            tracks = ExperimentalTracks.objects(track_length__in=steps_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, immobile=False)
    else:
        tracks = ExperimentalTracks.objects(track_length__in=steps_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, l1_classified_as='fBm', immobile=False)

    classification_accuracy = []
    for track in tracks:
        if net_name == 'L1 Network':
            classification_accuracy.append(track.l1_error)
        elif net_name == 'L2 Network':
            classification_accuracy.append(track.l2_error)
        else: 
            raise ValueError
    
    
    lower_x = np.percentile(classification_accuracy, 5)
    # histogram(classification_accuracy, c='orange', white_above=lower_x)
    # plt.axvline(x=lower_x,c='black',alpha=0.7,linestyle='dotted')
    # plt.ylabel('Frequency', fontsize=16)
    # plt.xlabel('Classification accuracy', fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.show()
    return lower_x

def show_classification_results(tl_range, exp_label, net_name):
    aux = 0
    for exp_cond in EXPERIMENTAL_CONDITIONS:
        # Get classification error
        pc = 1 - get_classification_error(tl_range,exp_label, exp_cond, net_name)

        if net_name == 'L1 Network':
            tracks = ExperimentalTracks.objects(track_length__in=tl_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, immobile=False)
        else:
            tracks = ExperimentalTracks.objects(track_length__in=tl_range, labeling_method=exp_label,
                                                experimental_condition=exp_cond, l1_classified_as='fBm', immobile=False)

        # Count each category
        count = [0 for i in L1_output_categories_labels]
        for track in tracks:
            if net_name == 'L1 Network':
                for i in range(len(L1_output_categories_labels)):
                    if track.l1_classified_as == L1_output_categories_labels[i]:
                        count[i] += 1
            else:
                for i in range(len(L2_output_categories_labels)):
                    if track.l2_classified_as == L2_output_categories_labels[i]:
                        count[i] += 1

        # Compute error limits
        count_n = count
        error_y0 = (100 * pc * count[0]/len(tracks), 100 * pc * (count[1]+count[2])/len(tracks)) 
        error_y1 = (100 * pc * count[1]/len(tracks), 100 * pc * (count[0]+count[2])/len(tracks)) 
        error_y2 = (100 * pc * count[2]/len(tracks), 100 * pc * (count[0]+count[1])/len(tracks)) 
        count = [(100 * x) / len(tracks) for x in count]
        error_y = [[error_y0[0], error_y1[0], error_y2[0]],[error_y0[1], error_y1[1], error_y2[1]]]

        # For data tables
        print('Network:{}, label:{}, condition:{}'.format(net_name, exp_label, exp_cond))
        if net_name == 'L1 Network':
            print('{}, {}, {}'.format(L1_output_categories_labels[0], L1_output_categories_labels[1], L1_output_categories_labels[2]))
        else:
            print('{}, {}, {}'.format(L2_output_categories_labels[0], L2_output_categories_labels[1], L2_output_categories_labels[2]))
        print('{}, {}, {}'.format(count[0], count[1], count[2]))
        print(count_n)
        print(error_y)
        
        # Plot bars
        plt.bar(x=[(aux + i) for i in range(3)], height=count, width=0.6, align='center',
                color=['firebrick', 'orangered', 'dodgerblue'], yerr=error_y)
        aux += 5

    plt.gca().axes.set_xticks([i for i in range(13)])
    plt.gca().axes.set_xticklabels(['', 'Control', '', '', '', '', 'CDx-Chol', '', '', '', '', 'CDx', ''], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('%', fontsize=16)
    if exp_label == 'BTX':
        exp_label = 'BTX'
    plt.title(exp_label,fontsize=16)


    if net_name == 'L1 Network':
        colors = {L1_output_categories_labels[0]: 'firebrick', L1_output_categories_labels[1]: 'orangered',
                  L1_output_categories_labels[2]: 'dodgerblue'}
    else:
        colors = {L2_output_categories_labels[0]: 'firebrick', L2_output_categories_labels[1]: 'orangered',
                  L2_output_categories_labels[2]: 'dodgerblue'}

    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    
    if net_name == 'L1 Network' and label == 'mAb':
        plt.legend(handles, ['fBm', 'CTRW', 'two-state'], bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=14)
    elif net_name == 'L2 Network' and label == 'mAb':
        plt.legend(handles, ['fBm subdiffusive', 'fBm Brownian', 'fBm superdiffusive'], bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=14)

    plt.rcParams['lines.color'] = 'b'
    plt.rcParams['lines.linewidth'] = 3
    plt.show()
"""

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')
all_trajectories = [trajectory for trajectory in Trajectory.objects() if 'prediction' in trajectory.info]
DatabaseHandler.disconnect()

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]

        predictions = [trajectory.info['prediction']['classified_model'] for trajectory in filtered_trajectories]
        classification_accuracies = [trajectory.info['prediction']['model_classification_accuracy'] for trajectory in filtered_trajectories]

        number_of_tracks = len(predictions)

        model_strings = [class_model.STRING_LABEL for class_model in ALL_MODELS]
        count = np.zeros((len(model_strings))).tolist()
        pc = 1 - np.percentile(classification_accuracies, 5)

        counter = Counter(predictions)

        for model_string in model_strings:
            count[model_strings.index(model_string)] = counter[model_string]

        errors = [[], []]

        for i in range(len(count)):
            error_yi = (100 * pc * count[i]/number_of_tracks, 100 * pc * np.sum(count[:i] + count[i+1:])/number_of_tracks) 
            errors[0].append(error_yi[0])
            errors[1].append(error_yi[1])

        count = [(100 * x) / number_of_tracks for x in count]

        with open(f"model_classification_{label}_{experimental_condition}.txt", 'w') as a_file:
            for model_string in model_strings:
                index = model_strings.index(model_string)
                a_file.write(f"{count[index]},{errors[1][index]},{errors[0][index]},")

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]
        filtered_trajectories = [trajectory for trajectory in filtered_trajectories if trajectory.info['prediction']['classified_model'] not in ['id', 'od']]

        predictions = [trajectory.info['prediction']['hurst_exponent'] for trajectory in filtered_trajectories]

        with open(f"hurst_exponent_{label}_{experimental_condition}.txt", 'w') as a_file:
            for prediction in predictions:
                a_file.write(f"{prediction}\n")

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]
        filtered_trajectories = [trajectory for trajectory in filtered_trajectories if trajectory.info['prediction']['classified_model'] == 'fbm']

        predictions = [trajectory.info['prediction']['diffusion_coefficient'] for trajectory in filtered_trajectories]

        with open(f"diffusion_coefficient_{label}_{experimental_condition}.txt", 'w') as a_file:
            for prediction in predictions:
                a_file.write(f"{prediction}\n")

for theoretical_model in ['fbm', 'sbm']:
    reference_models = SBM_MODELS if theoretical_model == 'sbm' else FBM_MODELS
    for label in ['BTX', 'mAb']:
        for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
            filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]
            filtered_trajectories = [trajectory for trajectory in filtered_trajectories if trajectory.info['prediction']['classified_model'] == theoretical_model]

            predictions = [trajectory.info['prediction']['sub_classified_model'] for trajectory in filtered_trajectories]
            classification_accuracies = [trajectory.info['prediction']['sub_model_classification_accuracy'] for trajectory in filtered_trajectories]

            number_of_tracks = len(predictions)

            model_strings = [class_model.STRING_LABEL for class_model in reference_models]
            count = np.zeros((len(model_strings))).tolist()
            pc = 1 - np.percentile(classification_accuracies, 5)

            counter = Counter(predictions)

            for model_string in model_strings:
                count[model_strings.index(model_string)] = counter[model_string]

            errors = [[], []]

            for i in range(len(count)):
                error_yi = (100 * pc * count[i]/number_of_tracks, 100 * pc * np.sum(count[:i] + count[i+1:])/number_of_tracks) 
                errors[0].append(error_yi[0])
                errors[1].append(error_yi[1])

            count = [(100 * x) / number_of_tracks for x in count]

            with open(f"sub_{theoretical_model}_model_classification_{label}_{experimental_condition}.txt", 'w') as a_file:
                for model_string in model_strings:
                    index = model_strings.index(model_string)
                    a_file.write(f"{count[index]},{errors[1][index]},{errors[0][index]},")
