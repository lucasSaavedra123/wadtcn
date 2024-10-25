import numpy as np

from DatabaseHandler import DatabaseHandler
from TheoreticalModels import ALL_MODELS, SBM_MODELS, FBM_MODELS
from Trajectory import Trajectory
from collections import Counter


DatabaseHandler.connect_over_network(None, None, '192.168.0.174', 'anomalous_diffusion_analysis')
all_trajectories = [trajectory for trajectory in Trajectory.objects() if 'prediction' in trajectory.info]
DatabaseHandler.disconnect()

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]

        predictions = [trajectory.info['prediction']['classified_model'] for trajectory in filtered_trajectories]
        classification_accuracies = [trajectory.info['prediction']['model_classification_accuracy'] for trajectory in filtered_trajectories]

        number_of_tracks = len(predictions)

        model_strings = ['attm', 'sbm', 'fbm', 'ctrw', 'lw', 'od', 'id']
        count = np.zeros((len(model_strings))).tolist()
        pc = 1 - np.percentile(classification_accuracies, 5)

        counter = Counter(predictions)

        for model_string in model_strings:
            count[model_strings.index(model_string)] = counter[model_string]

        errors = [[], []]

        for i in range(len(count)):
            error_yi = (100 * pc * count[i]/number_of_tracks, 100 * pc * (number_of_tracks - count[i])/number_of_tracks) 
            errors[0].append(error_yi[0])
            errors[1].append(error_yi[1])

        count = [(100 * x) / number_of_tracks for x in count]

        with open(f"model_classification_{label}_{experimental_condition}.txt", 'w') as a_file:
            for model_string in model_strings:
                index = model_strings.index(model_string)
                a_file.write(f"{count[index]},{count[index] + errors[1][index]},{count[index] - errors[0][index]},")

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
                error_yi = (100 * pc * count[i]/number_of_tracks, 100 * pc * (number_of_tracks - count[i])/number_of_tracks) 
                errors[0].append(error_yi[0])
                errors[1].append(error_yi[1])

            count = [(100 * x) / number_of_tracks for x in count]

            with open(f"sub_{theoretical_model}_model_classification_{label}_{experimental_condition}.txt", 'w') as a_file:
                for model_string in model_strings:
                    index = model_strings.index(model_string)
                    a_file.write(f"{count[index]},{count[index] + errors[1][index]},{count[index] - errors[0][index]},")

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        for theoretical_model in ['lw', 'ctrw', 'fbm', 'sbm', 'attm']:
            filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]
            filtered_trajectories = [trajectory for trajectory in filtered_trajectories if trajectory.info['prediction']['classified_model'] == theoretical_model]

            predictions = [trajectory.info['prediction']['hurst_exponent'] * 2 for trajectory in filtered_trajectories]

            with open(f"anomalous_exponents_{label}_{experimental_condition}_{theoretical_model}.txt", 'w') as a_file:
                for prediction in predictions:
                    a_file.write(f"{prediction}\n")

for label in ['BTX', 'mAb']:
    for experimental_condition in ['Control', 'CDx', 'CDx-Chol']:
        filtered_trajectories = [trajectory for trajectory in all_trajectories if trajectory.info['experimental_condition'] == experimental_condition and trajectory.info['label'] == label]
        filtered_trajectories = [trajectory for trajectory in filtered_trajectories if 'prediction' in trajectory.info and trajectory.info['prediction']['classified_model'] == 'id']

        with open(f"segment_residence_time_{label}_{experimental_condition}.txt", 'w') as a_file:
            for t in filtered_trajectories:
                t_segmentation = np.array(t.info['prediction']['segmentation'])[:,0]
                ts = t.sub_trajectories_trajectories_from_confinement_states(states=t_segmentation)

                for sub_t in ts[1]:
                    if sub_t.duration != 0:
                        a_file.write(f"{sub_t.duration}\n")
