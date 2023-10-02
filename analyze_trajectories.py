import tqdm
import matplotlib.pyplot as plt
import numpy as np

from DatabaseHandler import DatabaseHandler
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from DataSimulation import CustomDataSimulation
from CONSTANTS import EXPERIMENT_TIME_FRAME_BY_FRAME, IMMOBILE_THRESHOLD
from TheoreticalModels import ALL_MODELS, Model
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

all_trajectories = Trajectory.objects()

number_of_trayectories = len(all_trajectories)

filtered_trajectories = [trajectory for trajectory in all_trajectories if not trajectory.is_immobile(IMMOBILE_THRESHOLD)]

trajectories_by_length = defaultdict(lambda: [])

for trajectory in filtered_trajectories:
    trajectories_by_length[trajectory.length].append(trajectory)

number_of_immobile_trajectories = number_of_trayectories - len(filtered_trajectories)

print(f"There are {number_of_trayectories} trajectories and {number_of_immobile_trajectories} are immobile ({100 * round(number_of_immobile_trajectories/number_of_trayectories, 2)}%).")

reference_network = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trajectory_length=25, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())
assert len(reference_network) == 1
network_and_length = {25: reference_network[0]}
network_and_length[25].enable_database_persistance()
network_and_length[25].load_as_file()

classification_accuracies = []

for network in WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()):
    if network.trajectory_length != 25:
        network.set_wadnet_tcn_encoder(network_and_length[25], -4)
        network.enable_database_persistance()
        network.load_as_file()
        network_and_length[network.trajectory_length] = network

DatabaseHandler.disconnect()

predictions = []

number_of_tracks = 0

for length in tqdm.tqdm(trajectories_by_length.keys()):
    print(length)
    if length in network_and_length:
        trajectories = trajectories_by_length[length]
        classification_accuracies += [network_and_length[length].history_training_info['val_categorical_accuracy'][-1]] * len(trajectories)
        predictions += [ ALL_MODELS[i].STRING_LABEL for i in network_and_length[length].predict(trajectories).tolist()]
        number_of_tracks+=len(trajectories)

print(f"{number_of_tracks} trajectories were analyzed from {len(filtered_trajectories)} ({100 * round(number_of_tracks/len(filtered_trajectories), 2)}).")

model_strings = [class_model.STRING_LABEL for class_model in ALL_MODELS]
count = np.zeros((len(model_strings))).tolist()
pc = 1 - np.percentile(classification_accuracies, 5)

counter = Counter(predictions)

for model_string in model_strings:
    count[model_strings.index(model_string)] = counter[model_string]

errors = [[], []]
aux = 0

for i in range(len(count)):
    error_yi = (100 * pc * count[i]/number_of_tracks, 100 * pc * np.sum(count[:i] + count[i+1:])/number_of_tracks) 
    errors[0].append(error_yi[0])
    errors[1].append(error_yi[1])

count = [(100 * x) / number_of_tracks for x in count]

colors = [Model.Model.MODEL_COLORS[model_string] for model_string in model_strings]

plt.bar(x=[(aux + i) for i in range(len(ALL_MODELS))], height=count, width=0.6, align='center', color=colors, yerr=errors)
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
plt.legend(handles, model_strings, bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=14)

#plt.rcParams['lines.color'] = 'b'
#plt.rcParams['lines.linewidth'] = 3
plt.show()
