import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
import seaborn as sns
import umap

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation, CustomDataSimulation
from TheoreticalModels import ALL_MODELS, ANDI_MODELS
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier


def get_encoder_from_classifier(a_classifier):
    encoding_layer = a_classifier.architecture.layers[-4]
    encoding_model = Model(
        inputs=a_classifier.architecture.input,
        outputs=encoding_layer.output
    )

    return encoding_model

reference_lengths = [25,500,975]
simulator_reference = CustomDataSimulation

length_to_encoder = {}

print(f"Obtaining reference models...")
DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')

for reference_length in reference_lengths:
    already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(
        simulator_identifier=AndiDataSimulation.STRING_LABEL,
        trained=True,
        hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()
    )
    reference_classifier = [network for network in already_trained_networks if network.trajectory_length == reference_length][0]
    reference_classifier.enable_database_persistance()
    reference_classifier.load_as_file()
    reference_classifier.simulator = simulator_reference
    reference_encoder = get_encoder_from_classifier(reference_classifier)

    for length in reference_lengths:
        print(reference_length, "->", length)

        length_classifier = WaveNetTCNTheoreticalModelClassifier(length,length, simulator=simulator_reference)
        length_classifier.build_network()
        length_encoder = get_encoder_from_classifier(length_classifier)

        print("Transfering...")
        length_encoder.set_weights(reference_encoder.get_weights())

        print("Simulating Trajectories...")
        trajectories = simulator_reference().simulate_trajectories_by_model(12500, length, 12500, ALL_MODELS if simulator_reference == CustomDataSimulation else ANDI_MODELS)
        trajectories_transformed = length_classifier.transform_trajectories_to_input(trajectories)
        trajectories_labels = np.argmax(length_classifier.transform_trajectories_to_output(trajectories),axis=1)
        print("Encoding Trajectories...")
        trajectories_encoded = length_encoder.predict(trajectories_transformed)

        embedding = umap.UMAP().fit_transform(trajectories_encoded)

        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=1,
            c=[sns.color_palette()[x] for x in trajectories_labels]
        )
        plt.gca().set_aspect('equal', 'datalim')
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"{reference_length}_{length}_{simulator_reference.STRING_LABEL}.png")
        plt.clf()

DatabaseHandler.disconnect()
