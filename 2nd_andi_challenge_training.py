from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNModelSingleLevelPredicter import WavenetTCNModelSingleLevelPredicter
from PredictiveModel.WavenetTCNHurstExponentSingleLevelPredicter import WavenetTCNHurstExponentSingleLevelPredicter
from PredictiveModel.WavenetTCNDiffusionCoefficientSingleLevelPredicter import WavenetTCNDiffusionCoefficientSingleLevelPredicter
from PredictiveModel.WavenetTCNChangePointSingleLevelPredicter import WavenetTCNChangePointSingleLevelPredicter
from DatabaseHandler import DatabaseHandler


DatabaseHandler.connect_over_network(None, None, 'localhost', 'anomalous_diffusion')

import pandas as pd
from collections import Counter

#a = pd.read_csv('t_val_200_12500.cache')
#print(Counter(a['state_t']))
#exit()


"""
dataframe = pd.read_csv("t_val_200_12500.cache")

for a in dataframe['state_t'].unique():
   print(a, len(dataframe[dataframe['state_t'] == a])/len(dataframe))

exit()
"""
L = 200

networks = [
    #WavenetTCNChangePointSingleLevelPredicter,
    WavenetTCNModelSingleLevelPredicter, 
    WavenetTCNHurstExponentSingleLevelPredicter,
    WavenetTCNDiffusionCoefficientSingleLevelPredicter,
]


for network_index, network in enumerate(networks):
    #network.analyze_hyperparameters(L,L,initial_epochs=1,steps=1,simulator=Andi2ndDataSimulation)
    network = network(L, L, simulator=Andi2ndDataSimulation)
    #network.hyperparameters['epochs'] = 10
    #network.fit()
    #network.save_as_file()
    network.load_as_file()
    #network.plot_single_level_prediction()
    network.plot_confusion_matrix()
    exit()

DatabaseHandler.disconnect()