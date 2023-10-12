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

trained_networks = list(WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=CustomDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()))
trained_networks = sorted(trained_networks, key=lambda net: (net.trajectory_length, -net.trajectory_time))

for index, network in enumerate(trained_networks):
    if index == 0:
        reference_network = network   
    else:
        network.set_wadnet_tcn_encoder(reference_network, -4)

    network.enable_database_persistance()
    network.load_as_file()

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]
        filtered_trajectories = [trajectory for trajectory in filtered_trajectories if not trajectory.is_immobile(IMMOBILE_THRESHOLD) and trajectory.length >= 25]

        predictions = []

        for trajectory in tqdm.tqdm(filtered_trajectories):
            available_networks = [network for network in trained_networks if network.trajectory_length == trajectory.length and (network.trajectory_time * 0.85 <= trajectory.duration <= network.trajectory_time * 1.15)]

            if len(available_networks) == 0:
                continue
            elif len(available_networks) == 1:
                network = available_networks[0]
            else:
                network_to_select_index = np.argmin(np.abs(np.array([network.trajectory_time for network in available_networks]) - trajectory.duration))
                network = available_networks[network_to_select_index]

            trajectory.info['prediction'] = {
                'classified_model': [ ALL_MODELS[i].STRING_LABEL for i in network.predict([trajectory]).tolist()][0],
                'model_classification_accuracy': network.history_training_info['val_categorical_accuracy'][-1],
            }

            trajectory.save()

DatabaseHandler.disconnect()
