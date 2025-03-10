import keras_tuner as kt
from tensorflow import device, config

class PredictiveModelTuner(kt.HyperModel):
    def __init__(self, network_object):
        super().__init__()
        self.network_object = network_object

    def build(self, hp):
        return self.network_object.build_network(hp)

    def fit(self, hp, model, *args, **kwargs):
        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'
        with device(device_name):
            return model.fit(
                *args,
                **kwargs,
            )