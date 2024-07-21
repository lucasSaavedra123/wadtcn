import keras_tuner as kt

class PredictiveModelTuner(kt.HyperModel):
    def __init__(self, network_object, batch_size_values):
        super().__init__()
        self.network_object = network_object
        self.batch_size_values = batch_size_values

    def build(self, hp):
        return self.network_object.build_network(hp)

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", self.batch_size_values),
            **kwargs,
        )