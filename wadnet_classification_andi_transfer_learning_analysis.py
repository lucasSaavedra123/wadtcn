from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation, CustomDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.model_utils import get_encoder_from_classifier

f1_scores = []


#DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion_models')
#already_trained_networks = WaveNetTCNTheoreticalModelClassifier.objects(simulator_identifier=AndiDataSimulation.STRING_LABEL, trained=True, hyperparameters=WaveNetTCNTheoreticalModelClassifier.selected_hyperparameters())
#longest_length_classifier = [network for network in already_trained_networks if network.trajectory_length == 25][0]
#longest_length_classifier.enable_database_persistance()
#longest_length_classifier.load_as_file()
#DatabaseHandler.disconnect()

longest_length_classifier = WaveNetTCNTheoreticalModelClassifier(25,25 * 0.01, simulator=CustomDataSimulation)
longest_length_classifier.enable_early_stopping()
longest_length_classifier.fit()
#longest_length_classifier.load_as_file()
print(longest_length_classifier.plot_confusion_matrix())

longest_length_encoder = get_encoder_from_classifier(longest_length_classifier, -4)

shortest_length_classifier = WaveNetTCNTheoreticalModelClassifier(100,100 * 0.01, simulator=CustomDataSimulation)
shortest_length_classifier.build_network()
shortest_length_encoder = get_encoder_from_classifier(shortest_length_classifier, -4)

shortest_length_encoder.set_weights(longest_length_encoder.get_weights())

shortest_length_classifier.wadnet_tcn_encoder = shortest_length_encoder
shortest_length_classifier.build_network()

#shortest_length_classifier.hyperparameters['epochs'] = 1
shortest_length_classifier.enable_early_stopping()
shortest_length_classifier.fit()
print(shortest_length_classifier.plot_confusion_matrix())
