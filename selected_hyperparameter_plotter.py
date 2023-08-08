from DatabaseHandler import DatabaseHandler
from DataSimulation import AndiDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier
from PredictiveModel.WaveNetTCNFBMModelClassifier import WaveNetTCNFBMModelClassifier
from PredictiveModel.WaveNetTCNSBMModelClassifier import WaveNetTCNSBMModelClassifier
from PredictiveModel.WavenetTCNWithLSTMHurstExponentPredicter import WavenetTCNWithLSTMHurstExponentPredicter

DatabaseHandler.connect_over_network(None, None, '10.147.20.1', 'anomalous_diffusion')

"""

{
    <class 'PredictiveModel.WaveNetTCNTheoreticalModelClassifier.WaveNetTCNTheoreticalModelClassifier'>: {'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30, 'lr': 0.001}
    <class 'PredictiveModel.WaveNetTCNFBMModelClassifier.WaveNetTCNFBMModelClassifier'>: {'batch_size': 8, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30, 'lr': 0.0001}
    <class 'PredictiveModel.WaveNetTCNSBMModelClassifier.WaveNetTCNSBMModelClassifier'>: {'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 30, 'lr': 0.001}
    <class 'TheoreticalModels.ScaledBrownianMotion.ScaledBrownianMotionSubDiffusive'>: {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 30},
    <class 'TheoreticalModels.ScaledBrownianMotion.ScaledBrownianMotionBrownian'>: {'lr': 0.001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-07, 'epochs': 30},
    <class 'TheoreticalModels.ScaledBrownianMotion.ScaledBrownianMotionSuperDiffusive'>: {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 30},
    <class 'TheoreticalModels.FractionalBrownianMotion.FractionalBrownianMotionSubDiffusive'>: {'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 30},
    <class 'TheoreticalModels.FractionalBrownianMotion.FractionalBrownianMotionBrownian'>: {'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 30},
    <class 'TheoreticalModels.FractionalBrownianMotion.FractionalBrownianMotionSuperDiffusive'>: {'lr': 0.0001, 'batch_size': 8, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 30},
    <class 'TheoreticalModels.LevyWalk.LevyWalk'>: {'lr': 0.001, 'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 30}, 
    <class 'TheoreticalModels.ContinuousTimeRandomWalk.ContinuousTimeRandomWalk'>: {'lr': 0.0001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 30}, 
    <class 'TheoreticalModels.AnnealedTransientTimeMotion.AnnealedTransientTimeMotion'>: {'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30}}

"""

WaveNetTCNTheoreticalModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30, 'lr': 0.001}, title='First Classifier')
WaveNetTCNFBMModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={'batch_size': 8, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30, 'lr': 0.0001}, title='FBM Submodel Classifier')
WaveNetTCNSBMModelClassifier.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, discriminator={'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 30, 'lr': 0.001}, title='SBM Submodel Classifier')

WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='sbm_sub', discriminator={'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 30}, title='Anomalous Exponent of Subdiffusive SBM Predictor')
WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='sbm_brownian', discriminator={'lr': 0.001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-07, 'epochs': 30}, title='Anomalous Exponent of Brownian SBM Predictor')
WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='sbm_sup', discriminator={'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 30}, title='Anomalous Exponent of Supdiffusive SBM Predictor')

WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='fbm_sub', discriminator={'lr': 0.0001, 'batch_size': 64, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 30}, title='Anomalous Exponent of Subdiffusive FBM Predictor')
WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='fbm_brownian', discriminator={'lr': 0.001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 30}, title='Anomalous Exponent of Brownian FBM Predictor')
WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='fbm_sup', discriminator={'lr': 0.0001, 'batch_size': 8, 'amsgrad': False, 'epsilon': 1e-08, 'epochs': 30}, title='Anomalous Exponent of Supdiffusive FBM Predictor')

WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='lw', discriminator={'lr': 0.001, 'batch_size': 64, 'amsgrad': True, 'epsilon': 1e-06, 'epochs': 30}, title='Anomalous Exponent of LW Predictor')
WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='ctrw', discriminator={'lr': 0.0001, 'batch_size': 16, 'amsgrad': False, 'epsilon': 1e-06, 'epochs': 30}, title='Anomalous Exponent of CTRW Predictor')
WavenetTCNWithLSTMHurstExponentPredicter.plot_hyperparameter_search(25, 25, simulator=AndiDataSimulation, model='attm', discriminator={'lr': 0.0001, 'batch_size': 16, 'amsgrad': True, 'epsilon': 1e-08, 'epochs': 30}, title='Anomalous Exponent of ATTM Predictor')


DatabaseHandler.disconnect()
