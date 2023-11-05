from DataSimulation import CustomDataSimulation
from PredictiveModel.WaveNetTCNTheoreticalModelClassifier import WaveNetTCNTheoreticalModelClassifier

print(25)
network = WaveNetTCNTheoreticalModelClassifier(25,0.25,simulator=CustomDataSimulation)
network.enable_early_stopping()
network.fit()
network.plot_confusion_matrix()

print(50)
network = WaveNetTCNTheoreticalModelClassifier(50,0.50,simulator=CustomDataSimulation)
network.enable_early_stopping()
network.fit()
network.plot_confusion_matrix()

print(100)
network = WaveNetTCNTheoreticalModelClassifier(100,1,simulator=CustomDataSimulation)
network.enable_early_stopping()
network.fit()
network.plot_confusion_matrix()
