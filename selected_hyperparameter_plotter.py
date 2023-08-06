from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion')

WaveNetTCNTheoreticalModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30, 'lr': 0.001}, title='First Classifier')
WaveNetTCNFBMModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={'batch_size': 8, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30, 'lr': 0.0001}, title='FBM Submodel Classifier')
WaveNetTCNSBMModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 30, 'lr': 0.001}, title='SBM Submodel Classifier')

WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='sbm_sub', discriminator={'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 30}, title='Subdiffusive SBM Regresor')
WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='sbm_brownian', discriminator={'TODO': "Completar con los parámetros encontrados"}, title='Brownian SBM Regresor')

DatabaseHandler.disconnect()
