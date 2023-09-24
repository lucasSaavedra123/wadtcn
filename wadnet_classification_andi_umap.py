import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
import seaborn as sns
import umap

from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from TheoreticalModels import ANDI_MODELS
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier


def get_encoder_from_classifier(a_classifier):
    encoding_layer = a_classifier.architecture.layers[-4]
    encoding_model = Model(
        inputs=a_classifier.architecture.input,
        outputs=encoding_layer.output
    )

    return encoding_model

REFERENCE_LENGTH_ONE = 975
REFERENCE_LENGTH_TWO = 500

print(f"Obtaining reference models...")
DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')
already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(
    simulator_identifier=AndiDataSimulation.STRING_LABEL,
    trained=True,
    hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters()
)
reference_one_classifier = [network for network in already_trained_networks if network.trajectory_length == REFERENCE_LENGTH_ONE][0]
reference_one_classifier.enable_database_persistance()
reference_one_classifier.load_as_file()
DatabaseHandler.disconnect()

reference_length_one_encoder = get_encoder_from_classifier(reference_one_classifier)

reference_two_classifier = WaveNetTCNTheoreticalModelClassifier(REFERENCE_LENGTH_TWO,REFERENCE_LENGTH_TWO, simulator=AndiDataSimulation)
reference_two_classifier.build_network()
reference_length_two_encoder = get_encoder_from_classifier(reference_two_classifier)

reference_length_two_encoder.set_weights(reference_length_one_encoder.get_weights())

length_to_encoder = {
    REFERENCE_LENGTH_ONE: reference_length_one_encoder,
    REFERENCE_LENGTH_TWO: reference_length_two_encoder
}

length_to_classifier = {
    REFERENCE_LENGTH_ONE: reference_one_classifier,
    REFERENCE_LENGTH_TWO: reference_two_classifier
}

print("Encoding with lengths...")

for length in [REFERENCE_LENGTH_ONE, REFERENCE_LENGTH_TWO]:
    print(f"Simulating trajectories... (L={length})")
    trajectories = AndiDataSimulation().simulate_trajectories_by_model(12500, length, 12500, ANDI_MODELS)
    print(f"Converting trajectories... (L={length})")
    trajectories_transformed = length_to_classifier[length].transform_trajectories_to_input(trajectories)
    trajectories_labels = np.argmax(length_to_classifier[length].transform_trajectories_to_output(trajectories),axis=1)
    trajectories_encoded = length_to_encoder[length].predict(trajectories_transformed)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(trajectories_encoded)

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
    #plt.title('UMAP projection of the Penguin dataset', fontsize=24)
    plt.savefig(f"{REFERENCE_LENGTH_ONE}_{length}.png")
    plt.clf()
