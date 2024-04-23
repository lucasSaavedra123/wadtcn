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


for network_index, network in list(enumerate(networks))[2:]:
    #network.analyze_hyperparameters(L,L,initial_epochs=1,steps=1,simulator=Andi2ndDataSimulation)
    network = network(L, L, simulator=Andi2ndDataSimulation)
    
    try:
        network.load_as_file()
        if network_index == 0:
            network.plot_confusion_matrix()
        else:
            network.plot_single_level_prediction()
    except Exception as e:
        print(e)
        network.hyperparameters['epochs'] = 5
        network.fit()
        network.save_as_file()

DatabaseHandler.disconnect()