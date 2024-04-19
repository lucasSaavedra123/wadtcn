from DataSimulation import Andi2ndDataSimulation
from PredictiveModel.WavenetTCNModelSingleLevelPredicter import WavenetTCNModelSingleLevelPredicter
from PredictiveModel.WavenetTCNHurstExponentSingleLevelPredicter import WavenetTCNHurstExponentSingleLevelPredicter
from PredictiveModel.WavenetTCNDiffusionCoefficientSingleLevelPredicter import WavenetTCNDiffusionCoefficientSingleLevelPredicter


"""
dataframe = pd.read_csv("t_val_200_12500.cache")

for a in dataframe['state_t'].unique():
   print(a, len(dataframe[dataframe['state_t'] == a])/len(dataframe))

exit()
"""
L = 200

networks = [
    WavenetTCNModelSingleLevelPredicter(L, L, simulator=Andi2ndDataSimulation), 
    WavenetTCNHurstExponentSingleLevelPredicter(L, L, simulator=Andi2ndDataSimulation),
    WavenetTCNDiffusionCoefficientSingleLevelPredicter(L, L, simulator=Andi2ndDataSimulation),
]

for network_index, network in enumerate(networks[1:]):
    network.enable_early_stopping()
    network.fit()
    network.save_as_file()
    #network.load_as_file()
    network.plot_single_level_prediction()
    #network.plot_confusion_matrix()
    #exit()
