import keras_tuner as kt

class PredictiveModelTuner(kt.HyperModel):
    def __init__(self, network_object):
        super().__init__()
        self.network_object = network_object

    def build(self, hp):
        return self.network_object.build_network(hp)

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )