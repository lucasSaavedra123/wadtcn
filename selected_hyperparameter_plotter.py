from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion')

WaveNetTCNTheoreticalModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={
    'batch_size': 64,
    'amsgrad': True,
    'epsilon': 1e-08,
    'epochs': 30,
    'lr': 0.001
})

WaveNetTCNFBMModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={
    'batch_size': 8,
    'amsgrad': True,
    'epsilon': 1e-8,
    'epochs': 30,
    'lr': 0.0001
})

DatabaseHandler.disconnect()