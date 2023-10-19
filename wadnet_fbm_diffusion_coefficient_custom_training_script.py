import os
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
import tqdm
from DataSimulation import CustomDataSimulation
from PredictiveModel.WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter import WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter
from tensorflow.keras.backend import clear_session


DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_analysis')

raw_trajectories = list(Trajectory._get_collection().find({'info.prediction.classified_model': 'fbm'}, {}))

if not os.path.exists('lengths_and_durations_fbm_cache.txt'):
    lengths_durations = [(len(trajectory['x']), round(trajectory['t'][-1]-trajectory['t'][0],2)) for trajectory in raw_trajectories]
    lengths_durations = list(set(lengths_durations))
    lengths_durations = sorted(lengths_durations, key=lambda x: (x[0], -x[1]))

    with open('lengths_and_durations_fbm_cache.txt', 'w') as file:
        file.write('\n'.join(str(length_duration[0])+','+str(length_duration[1]) for length_duration in lengths_durations))
else:
    with open('lengths_and_durations_fbm_cache.txt', 'r') as file:
        lengths_durations = [(int(line.strip().split(',')[0]), float(line.strip().split(',')[1]))  for line in file.readlines()]

already_trained_networks = list(WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter.objects(
    simulator_identifier=CustomDataSimulation.STRING_LABEL,
    trained=True,
    hyperparameters=WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter.selected_hyperparameters()
))

for index, length_duration in tqdm.tqdm(list(enumerate(lengths_durations))):
    print("Length,Duration:", length_duration)
    length = length_duration[0]
    duration = length_duration[1]
    clear_session()

    available_networks = [network for network in already_trained_networks if network.trajectory_length == length and (network.trajectory_time * 0.85 <= duration <= network.trajectory_time * 1.15)]

    if len(available_networks) == 0:
        classifier = WavenetTCNWithLSTMDiffusionCoefficientFBMPredicter(length, duration, simulator=CustomDataSimulation)

        if index == 0:
            reference_architecture = classifier
        else:
            classifier.set_wadnet_tcn_encoder(reference_architecture, -3)

        classifier.enable_early_stopping()
        classifier.enable_database_persistance()
        classifier.fit()
        classifier.save()

        already_trained_networks.append(classifier)
    else:
        if index == 0:
            if len(available_networks) == 1:
                classifier = available_networks[0]
            else:
                classifier = max(*available_networks, key= lambda net: net.trajectory_time)

            reference_architecture = classifier
            classifier.enable_database_persistance()
            classifier.load_as_file()


DatabaseHandler.disconnect()