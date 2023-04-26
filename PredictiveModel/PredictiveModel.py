import pickle

from mongoengine import Document, IntField, FileField, DictField, FloatField, BooleanField, StringField
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import keras.backend as K

from DataSimulation import CustomDataSimulation, AndiDataSimulation


class TrackGenerator(Sequence):
    def __init__(self, batches, batch_size, dataset_function):
        self.batches = batches
        self.batch_size = batch_size
        self.dataset_function = dataset_function

    def __getitem__(self, item):
        tracks, classes = self.dataset_function(self.batch_size)
        return tracks, classes

    def __len__(self):
        return self.batches

class PredictiveModel(Document):
    #Must
    trajectory_length = IntField(min_value=1, required=True)
    trajectory_time = FloatField(required=True)
    hyperparameters = DictField(required=True)
    trained = BooleanField(default=False, required=True)
    simulator_identifier = StringField(required=True)

    #optional
    history_training_info = DictField(required=False)
    model_weights = FileField(required=False)
    extra_parameters = DictField(required=False)
    optimal_model = BooleanField(required=False)

    meta = {
        'allow_inheritance': True
    }

    @classmethod
    def analyze_hyperparameters(cls, trajectory_length, trajectory_time, initial_epochs=10, steps=10, **kwargs):
        # Stack names and lists position
        hyperparameters_to_analyze = cls.default_hyperparameters_analysis()

        if len(hyperparameters_to_analyze) > 0:
            stack_names = [k for k, v in hyperparameters_to_analyze.items()]
            stack = [0 for i in stack_names]
            tos = len(stack) - 1
            analysis_ended = False
            increasing = True

            # Compute and print number of combinations
            number_of_combinations = len(hyperparameters_to_analyze[stack_names[0]])
            for i in range(1, len(stack_names)):
                number_of_combinations *= len(hyperparameters_to_analyze[stack_names[i]])
            print("Total of combinations:{}".format(number_of_combinations))

            # Run the analysis
            while not analysis_ended:
                if tos == (len(stack) - 1) and stack[tos] < len(hyperparameters_to_analyze[stack_names[tos]]):
                    K.clear_session()
                    network = cls(trajectory_length, trajectory_time, **kwargs)
                    for i in range(len(stack_names)):
                        network.hyperparameters[stack_names[i]] = hyperparameters_to_analyze[stack_names[i]][stack[i]]
                    network.hyperparameters['epochs'] = initial_epochs
                    print('Evaluating params: {}'.format(network.hyperparameters))

                    #Check if this configuration it was already trained
                    if cls.objects(trajectory_length=trajectory_length, trajectory_time=trajectory_time, simulator_identifier=kwargs['simulator'].STRING_LABEL, hyperparameters=network.hyperparameters, trained=True).count() == 0:
                        network.fit()
                        network.enable_database_persistance()
                        network.save()

                    stack[tos] += 1
                elif tos == (len(stack) - 1) and stack[tos] == len(hyperparameters_to_analyze[stack_names[tos]]):
                    stack[tos] = 0
                    tos -= 1
                    increasing = False

                elif 0 < tos < (len(stack) - 1) and increasing:
                    tos += 1
                    increasing = True
                elif 0 < tos < (len(stack) - 1) and not increasing and stack[tos] + 1 <= len(
                        hyperparameters_to_analyze[stack_names[tos]]) - 1:
                    stack[tos] += 1
                    tos += 1
                    increasing = True
                elif 0 < tos < (len(stack) - 1) and not increasing and stack[tos] + 1 > len(
                        hyperparameters_to_analyze[stack_names[tos]]) - 1:
                    stack[tos] = 0
                    tos -= 1
                    increasing = False
                elif tos == 0 and not increasing and stack[tos] + 1 < len(hyperparameters_to_analyze[stack_names[tos]]):
                    stack[tos] += 1
                    tos += 1
                    increasing = True
                else:
                    analysis_ended = True        

        return cls.post_grid_search_analysis(trajectory_length, trajectory_time, initial_epochs, steps, **kwargs)

    @classmethod
    def post_grid_search_analysis(cls, trajectory_length, trajectory_time, current_epochs, step, **kwargs):
        networks = cls.objects(trajectory_length=trajectory_length, trajectory_time=trajectory_time, simulator_identifier=kwargs['simulator'].STRING_LABEL, trained=True)

        if len(networks) == 1:
            print(f"Hyperparameter Search Finished. Hyperparameters selected: {networks[0].hyperparameters}")
            return networks[0].hyperparameters

        networks = sorted(networks, key=lambda x: x.history_training_info['val_loss'][-1])
        networks = networks[:len(networks)//2]

        if len(networks) == 1:
            print(f"Hyperparameter Search Finished. Hyperparameters selected: {networks[0].hyperparameters}")
            return networks[0].hyperparameters

        print(f"Now will keep training train {len(networks)} networks...")

        for network in networks:
            network = cls(trajectory_length, trajectory_time, **kwargs)
            network.hyperparameters['epochs'] = current_epochs + step
            print('Evaluating params: {}'.format(network.hyperparameters))

            #Check if this configuration it was already trained
            if cls.objects(trajectory_length=trajectory_length, trajectory_time=trajectory_time, simulator_identifier=kwargs['simulator'].STRING_LABEL, hyperparameters=network.hyperparameters, trained=True).count() == 0:
                network.fit()
                network.enable_database_persistance()
                network.save()

        return cls.post_grid_search_analysis(trajectory_length, trajectory_time, current_epochs + step, step, **kwargs)


    def __init__(self, trajectory_length, trajectory_time, **kwargs):
        self.architecture = None
        self.hyperparameters_analysis = self.__class__.default_hyperparameters_analysis()
        self.db_persistance = False

        if 'simulator_identifier' in kwargs:
            simulator_identifier = kwargs['simulator_identifier']
            if simulator_identifier == 'custom':
                self.simulator = CustomDataSimulation
            elif simulator_identifier == 'andi':
                self.simulator = AndiDataSimulation
            else:
                Exception(f'simulator_identifier not recognized. It was {simulator_identifier}')
            
            del kwargs['simulator_identifier']
        else:
            self.simulator = kwargs['simulator']
            simulator_identifier = kwargs['simulator'].STRING_LABEL
            del kwargs['simulator']

        if 'hyperparameters' in kwargs:
            hyperparameters = kwargs['hyperparameters']
            del kwargs['hyperparameters']
        else:
            hyperparameters = self.default_hyperparameters()

        if 'id' in kwargs:
            super().__init__(
                trajectory_length=trajectory_length,
                trajectory_time=trajectory_time,
                hyperparameters=hyperparameters,
                simulator_identifier=simulator_identifier,
                **kwargs
            )
        else:
            super().__init__(
                trajectory_length=trajectory_length,
                trajectory_time=trajectory_time,
                hyperparameters=hyperparameters,
                simulator_identifier=simulator_identifier,
                extra_parameters = kwargs
            )

    def enable_database_persistance(self):
        self.db_persistance = True

    def disable_database_persistance(self):
        self.db_persistance = False

    def default_hyperparameters(self):
        raise Exception("default_hyperparameters should be defined")
    
    @classmethod
    def default_hyperparameters_analysis(self):
        raise Exception("default_hyperparameters_analysis should be defined")

    def plot_training_history(self):
        assert self.history_training_info is not None, "There is not training history"

        TRANSLATION = {
            'loss':'Loss',
            'mae' : 'MAE',
            'mse' : 'MSE',
            'val_loss': 'Validation Loss',
            'val_mae': 'Validation MAE',
            'categorical_accuracy': 'Accuracy',
            'val_categorical_accuracy': 'Validation Accuracy'
        }

        for metric in self.history_training_info.keys():
            epochs = range(1, len(self.history_training_info[metric])+1)

            if 'val' not in metric:
                plt.plot(epochs, self.history_training_info[metric])
                plt.plot(epochs, self.history_training_info['val_'+metric])
                plt.title(f"L={self.trajectory_length}, models={[model.STRING_LABEL for model in self.models_involved_in_predictive_model]}")
                plt.ylabel(TRANSLATION[metric])
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.grid()
                plt.show()

    def __str__(self):
        return f"{self.type_name}_{str(self.simulator().STRING_LABEL)}_{'_'.join([model.STRING_LABEL for model in self.models_involved_in_predictive_model])}"

    def prepare_dataset(self, set_size):
        trajectories = self.simulator().simulate_trajectories_by_model(set_size, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def save_as_file(self):
        if self.architecture is not None:
            if self.db_persistance:
                if self.model_weights is not None:
                    self.model_weights.replace(pickle.dumps(self.architecture.get_weights()))
                else:
                    self.model_weights.put(pickle.dumps(self.architecture.get_weights()))
            else:
                self.architecture.save_weights(f'{str(self)}.h5')
        else:
            print(f"As architecture is not defined, {self} architecture will not be persisted")

    def load_as_file(self):
        self.build_network()
        if self.db_persistance:
            weights = pickle.loads(self.model_weights.read())

            if weights is not None:
                self.architecture.set_weights(weights)
        else:        
            self.architecture.load_weights(f'{str(self)}.h5')

    def save(self):
        self.save_as_file()
        super().save()

    def fit(self):
        if not self.trained:
            self.build_network()
            real_epochs = self.hyperparameters['epochs']
        else:
            real_epochs = self.hyperparameters['epochs'] - len(self.history_training_info['loss'])

        self.architecture.summary()

        if self.hyperparameters['with_early_stopping']:
            callbacks = [EarlyStopping(
                monitor="val_loss",
                min_delta=1e-3,
                patience=5,
                verbose=1,
                mode="min")]
        else:
            callbacks = []

        try:
            history_training_info = self.architecture.fit(
                TrackGenerator(self.hyperparameters['training_set_size']//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.prepare_dataset),
                epochs=real_epochs,
                callbacks=callbacks,
                validation_data=TrackGenerator(self.hyperparameters['validation_set_size']//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.prepare_dataset), shuffle=True
            ).history
        except KeyError:
            history_training_info = self.architecture.fit(
                TrackGenerator(self.hyperparameters['training_set_size']//self.hyperparameters[self.extra_parameters['model']]['batch_size'], self.hyperparameters[self.extra_parameters['model']]['batch_size'], self.prepare_dataset),
                epochs=real_epochs,
                callbacks=callbacks,
                validation_data=TrackGenerator(self.hyperparameters['validation_set_size']//self.hyperparameters[self.extra_parameters['model']]['batch_size'], self.hyperparameters[self.extra_parameters['model']]['batch_size'], self.prepare_dataset), shuffle=True
            ).history

        if self.trained:
            for dict_key in history_training_info:
                self.history_training_info[dict_key] += history_training_info[dict_key]
        else:
            self.history_training_info = history_training_info
            self.trained = True
