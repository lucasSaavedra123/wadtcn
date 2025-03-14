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
delta_t = 0.01 if simulator_reference.STRING_LABEL == 'andi' else 1

length_to_encoder = {}

print(f"Obtaining reference models...")
DatabaseHandler.connect_over_network(None, None, '192.168.0.174', 'anomalous_diffusion_analysis')
#DatabaseHandler.connect_over_network(None, None, '192.168.0.174', 'anomalous_diffusion')

for reference_length in reference_lengths:
    already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(
        simulator_identifier=simulator_reference.STRING_LABEL,
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

        length_classifier = WaveNetTCNTheoreticalModelClassifier(length,length*delta_t, simulator=simulator_reference)
        length_classifier.build_network()
        length_encoder = get_encoder_from_classifier(length_classifier)

        print("Transfering...")
        length_encoder.set_weights(reference_encoder.get_weights())

        print("Simulating Trajectories...")
        trajectories = simulator_reference().simulate_trajectories_by_model(12_500, length, length * delta_t, ALL_MODELS if simulator_reference == CustomDataSimulation else ANDI_MODELS)
        trajectories_transformed = length_classifier.transform_trajectories_to_input(trajectories)
        trajectories_labels = np.argmax(length_classifier.transform_trajectories_to_output(trajectories),axis=1)
        print("Encoding Trajectories...")
        trajectories_encoded = length_encoder.predict(trajectories_transformed)

        embedding = umap.UMAP().fit_transform(trajectories_encoded)

        fig, ax = plt.subplots()

        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=1,
            c=[sns.color_palette()[x] for x in trajectories_labels]
        )
        #plt.gca().set_aspect('equal', 'datalim')
        #plt.tight_layout()
        ax.set_xlabel('UMAP 1',fontsize=20)
        ax.set_ylabel('UMAP 2',fontsize=20)
        ax.set_box_aspect(1)

        fig.subplots_adjust(
            top=0.835,
            bottom=0.155,
            left=0.125,
            right=0.9,
            hspace=0.2,
            wspace=0.2
        )

        #plt.xticks([0,5,10,15])
        #plt.yticks([0,5,10,15])
        plt.tick_params(axis='both', labelsize=20)
        plt.savefig(f"{reference_length}_{length}_{simulator_reference.STRING_LABEL}.jpg", dpi=600)
        #plt.show()
        plt.clf()

DatabaseHandler.disconnect()
