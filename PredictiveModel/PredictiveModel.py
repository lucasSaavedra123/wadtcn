from os.path import join

import pickle
from threading import Thread, Event
from queue import Queue, Full

from mongoengine import Document, IntField, FileField, DictField, FloatField, BooleanField, StringField
from keras.callbacks import EarlyStopping, Callback
from tensorflow import device, config
import matplotlib.pyplot as plt
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error
import matplotlib.patches as mpatches

from CONSTANTS import TRAINING_SET_SIZE_PER_EPOCH, VALIDATION_SET_SIZE_PER_EPOCH, NETWORKS_DIRECTORY
from TheoreticalModels import ALL_MODELS, ANDI_MODELS
from DataSimulation import CustomDataSimulation, AndiDataSimulation, Andi2ndDataSimulation
from .model_utils import ThreadedTrackGenerator, TrackGenerator, get_encoder_from_classifier

class CustomCallback(Callback):
    def __init__(self, thread_queue):
        super().__init__()
        self.queue = thread_queue

    def on_epoch_end(self, epoch, logs=None):
        print(f" INFO: Thread Queue Length: {self.queue.qsize()}")


def tool_equal_dicts(d1, d2, ignore_keys):
    d1_filtered = {k:v for k,v in d1.items() if k not in ignore_keys}
    d2_filtered = {k:v for k,v in d2.items() if k not in ignore_keys}
    return d1_filtered == d2_filtered

def tool_include_classifier(current_hyperparameter, old_hyperparameter):
    return tool_equal_dicts(old_hyperparameter, current_hyperparameter, ['epochs'])

def generate_new_dictionary(dictionary):
    return {k:v for k,v in dictionary.items()}

def generate_colors_for_hyperparameters_list(hyperparameter_values):
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'olive', 'cyan', 'black', 'magenta', 'navy', 'lime', 'yellow']
    
    hyperparameter_value_to_color = {}

    for index, hyperparameter_value in enumerate(hyperparameter_values):
        hyperparameter_value_to_color[hyperparameter_value] = color_list[index % len(color_list)]

    return hyperparameter_value_to_color

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
        """
        network_object = cls(trajectory_length, trajectory_time, **kwargs)

        tuner = kt.BayesianOptimization(
            PredictiveModelTuner(network_object, [50, 100]),
            objective='val_loss',
            max_trials=50)

        X, Y = network_object.prepare_dataset(100000)

        tuner.search(X, Y, epochs=5, validation_split=0.2)

        return None
        """
        # Stack names and lists position
        hyperparameters_to_analyze = cls.default_hyperparameters_analysis()

        networks_list = []

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

            combination_index = 1
            # Run the analysis
            while not analysis_ended:
                
                if tos == (len(stack) - 1) and stack[tos] < len(hyperparameters_to_analyze[stack_names[tos]]):
                    K.clear_session()
                    network = cls(trajectory_length, trajectory_time, **kwargs)
                    for i in range(len(stack_names)):
                        network.hyperparameters[stack_names[i]] = hyperparameters_to_analyze[stack_names[i]][stack[i]]

                    #Check if this configuration it was already trained
                    if 'model' in kwargs:
                        first_classifiers = [classifier for classifier in cls.objects(trajectory_length=trajectory_length, trajectory_time=trajectory_time, simulator_identifier=kwargs['simulator'].STRING_LABEL, trained=True) if tool_include_classifier(network.hyperparameters, classifier.hyperparameters)]
                        classifiers = [classifier for classifier in first_classifiers if classifier.extra_parameters['model'] == kwargs['model']]
                    else:
                        classifiers = [classifier for classifier in cls.objects(trajectory_length=trajectory_length, trajectory_time=trajectory_time, simulator_identifier=kwargs['simulator'].STRING_LABEL, trained=True) if tool_include_classifier(network.hyperparameters, classifier.hyperparameters)]

                    if len(classifiers) == 0:
                        K.clear_session()
                        network.hyperparameters['epochs'] = initial_epochs
                        print('{}) Evaluating params: {}'.format(combination_index, network.hyperparameters))
                        network.fit()
                        network.enable_database_persistance()
                        network.save()
                        networks_list.append(network)
                    elif len(classifiers) == 1:
                        print('{}) Evaluating params: {}'.format(combination_index, classifiers[-1].hyperparameters))
                        classifiers[-1].enable_database_persistance()
                        classifiers[-1].load_as_file()
                        networks_list.append(classifiers[-1])
                    else:
                        raise Exception(f'More than one classifier was returned ({len(classifiers)})')

                    combination_index += 1
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

        return cls.post_grid_search_analysis(networks_list, trajectory_length, trajectory_time, initial_epochs, steps, **kwargs)

    @classmethod
    def post_grid_search_analysis(cls, networks, trajectory_length, trajectory_time, current_epochs, step, **kwargs):
        if len(networks) == 1:
            print(f"Hyperparameter Search Finished. Hyperparameters selected: {networks[0].hyperparameters}")
            return networks[0].hyperparameters
        else:
            networks = sorted(networks, key=lambda x: x.history_training_info['val_loss'][current_epochs-1])
            networks = networks[:len(networks)//2]

            if len(networks) == 1:
                print(f"Hyperparameter Search Finished. Hyperparameters selected: {networks[0].hyperparameters}")
                return networks[0].hyperparameters

        print(f"Now will keep training train {len(networks)} networks...")

        for network in networks:
            #network = cls(trajectory_length, trajectory_time, **kwargs)
            network.enable_database_persistance()
            #network.load_as_file()

            print('Evaluating params: {}'.format(network.hyperparameters))
            
            if network.hyperparameters['epochs'] < current_epochs + step:
                network.hyperparameters['epochs'] = current_epochs + step
                network.fit()
                network.save()

        return cls.post_grid_search_analysis(networks, trajectory_length, trajectory_time, current_epochs + step, step, **kwargs)

    @classmethod
    def plot_hyperparameter_search(cls, trajectory_length, trajectory_time, discriminator=None, title=None, **kwargs):    
        max_epochs = float("-inf")

        if discriminator is not None and not type(discriminator) is dict:
            hyperparameter_values = cls.default_hyperparameters_analysis()[discriminator]
            hyperparameter_value_to_color = generate_colors_for_hyperparameters_list(hyperparameter_values)

        if 'model' not in kwargs:
            models_to_show = [d for d in cls.objects.all() if d.trajectory_length == trajectory_length and d.trajectory_time == trajectory_time and kwargs['simulator'] == d.simulator]
        else:
            models_to_show = [d for d in cls.objects.all() if d.trajectory_length == trajectory_length and d.trajectory_time == trajectory_time and kwargs['simulator'] == d.simulator and kwargs['model'] == d.extra_parameters['model']]

        for predictive_model in models_to_show:
            new_error = np.array(predictive_model.history_training_info['val_loss'])

            max_epochs = max(max_epochs, len(new_error))

            if discriminator is None:
                plt.plot(range(1, len(new_error)+1), new_error)
            elif type(discriminator) is dict:
                if discriminator == predictive_model.hyperparameters:
                    plt.plot(range(1, len(new_error)+1), new_error, color='red', zorder=999)
                else:
                    plt.plot(range(1, len(new_error)+1), new_error, color='grey')
            else:
                plt.plot(range(1, len(new_error)+1), new_error, color = hyperparameter_value_to_color[predictive_model.hyperparameters[discriminator]])

        if title is not None:
            plt.title(f"{title}")
        else:
            plt.title(f"L={trajectory_length}")

        plt.ylabel('Validation Loss')
        plt.xlim([1,max_epochs])

        plt.xlabel('Epoch')
        plt.grid()

        if discriminator is not None:
            if not type(discriminator) is dict:
                handles = []

                for hyperparameter_value in hyperparameter_values:
                    handles.append(mpatches.Patch(color=hyperparameter_value_to_color[hyperparameter_value], label=f"{discriminator}={hyperparameter_value}"))
            else:
                handles = [mpatches.Patch(color='red', label='Selected Hyperparameter'), mpatches.Patch(color='grey', label='Not Selected Hyperparameters')]

            plt.legend(handles=handles)

        plt.show()

    @classmethod
    def it_does_already_be_trained(cls, trajectory_length, trajectory_time, simulator, from_db=True):
        if from_db:
            classifiers = cls.objects(
                trajectory_length=trajectory_length,
                trajectory_time=trajectory_time,
                simulator_identifier=simulator.STRING_LABEL,
                trained=True
            )
            
            if len(classifiers) == 0:
                return False
            if len(classifiers) == 1:
                return True
            elif len(classifiers) > 1:
                raise Exception('There are more than one classifier persisted of the same characteristics')
        else:
            #return os.path.exists(f'{str(self)}.h5')
            return False

    def __init__(self, trajectory_length, trajectory_time, **kwargs):
        self.architecture = None
        self.hyperparameters_analysis = self.__class__.default_hyperparameters_analysis()
        self.db_persistance = False
        self.early_stopping = False
        self.wadnet_tcn_encoder = None

        if 'simulator_identifier' in kwargs:
            simulator_identifier = kwargs['simulator_identifier']
            if simulator_identifier == 'custom':
                self.simulator = CustomDataSimulation
            elif simulator_identifier == 'andi':
                self.simulator = AndiDataSimulation
            elif simulator_identifier == 'andi2':
                self.simulator = Andi2ndDataSimulation
            elif simulator_identifier is None:
                self.simulator = None
            else:
                raise Exception(f'simulator_identifier not recognized. It was {simulator_identifier}')
            
            del kwargs['simulator_identifier']
        else:
            self.simulator = kwargs['simulator']
            simulator_identifier = kwargs['simulator'].STRING_LABEL
            del kwargs['simulator']

        if 'hyperparameters' in kwargs:
            hyperparameters = kwargs['hyperparameters']
            del kwargs['hyperparameters']
        else:
            hyperparameters = self.default_hyperparameters(**kwargs)

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

    def set_wadnet_tcn_encoder(self, reference_classifier, layer_index):
        reference_encoder = get_encoder_from_classifier(reference_classifier, layer_index)

        self.build_network()
        self.wadnet_tcn_encoder = get_encoder_from_classifier(self, layer_index)
        self.wadnet_tcn_encoder.set_weights(reference_encoder.get_weights())

        self.architecture = None

    @property
    def number_of_models_involved(self):
        return len(self.models_involved_in_predictive_model)

    def enable_database_persistance(self):
        self.db_persistance = True

    def disable_database_persistance(self):
        self.db_persistance = False

    def enable_early_stopping(self):
        self.early_stopping = True

    def disable_early_stopping(self):
        self.early_stopping = False

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
            'val_categorical_accuracy': 'Validation Accuracy',
            'mean_absolute_percentage_error': 'MAPE',
            'val_mean_absolute_percentage_error': 'Validation MAPE',
        }

        for metric in self.history_training_info.keys():
            epochs = range(1, len(self.history_training_info[metric])+1)

            if metric in TRANSLATION and 'val' not in metric:
                plt.plot(epochs, self.history_training_info[metric])
                plt.plot(epochs, self.history_training_info['val_'+metric])
                try:
                    plt.title(f"L={self.trajectory_length}, models={[model.STRING_LABEL for model in self.models_involved_in_predictive_model]}")
                except AttributeError:
                    pass
                plt.ylabel(TRANSLATION[metric])
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.grid()
                plt.show()

    def __str__(self):
        return f"{self.type_name}_{self.trajectory_length}_{self.trajectory_time}_{self.simulator.STRING_LABEL}_{'_'.join([model.STRING_LABEL for model in self.models_involved_in_predictive_model])}"

    def prepare_dataset(self, set_size):
        trajectories = self.simulator().simulate_trajectories_by_model(set_size, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)
        return self.transform_trajectories_to_input(trajectories), self.transform_trajectories_to_output(trajectories)

    def model_to_label(self, model):
        return self.models_involved_in_predictive_model.index(model.__class__)

    def save_as_file(self):
        if self.architecture is not None:
            if self.db_persistance:
                if self.model_weights.get() is not None:
                    self.model_weights.replace(pickle.dumps(self.architecture.get_weights()))
                else:
                    self.model_weights.put(pickle.dumps(self.architecture.get_weights()))
            else:
                self.architecture.save_weights(join(NETWORKS_DIRECTORY, f"{str(self)}.h5"))
        else:
            print(f"As architecture is not defined, {self} architecture will not be persisted")

    def load_as_file(self):
        self.build_network()
        if self.db_persistance:
            weights = pickle.loads(self.model_weights.read())

            if weights is not None:
                self.architecture.set_weights(weights)
        else:
            self.architecture.load_weights(join(NETWORKS_DIRECTORY, f"{str(self)}.h5"))

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

        """
        trajectories_queue = Queue(maxsize=TRAINING_SET_SIZE_PER_EPOCH)
        finished_training_event = Event()
        """

        if self.early_stopping:
            callbacks = [EarlyStopping(
                monitor="val_loss",
                min_delta=1e-3,
                patience=5,
                verbose=1,
                mode="min")]
        else:
            callbacks = []

        """
        callbacks += [CustomCallback(trajectories_queue)]

        def create_work(queue, stop_event):
            while not stop_event.is_set():
                try:
                    queue.put(self.simulator().simulate_trajectories_by_model(1, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)[0], block=False)
                except Full:
                    pass
            print("STOPPING")

        producer = Thread(target=create_work, args=[trajectories_queue, finished_training_event], daemon=True)
        producer.start()
        """
        device_name = '/gpu:0' if len(config.list_physical_devices('GPU')) == 1 else '/cpu:0'

        if self.wadnet_tcn_encoder is None:
            with device(device_name):
                history_training_info = self.architecture.fit(
                    TrackGenerator(TRAINING_SET_SIZE_PER_EPOCH//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.prepare_dataset),
                    epochs=real_epochs,
                    callbacks=callbacks,
                    validation_data=TrackGenerator(VALIDATION_SET_SIZE_PER_EPOCH//self.hyperparameters['batch_size'], self.hyperparameters['batch_size'], self.prepare_dataset),
                    shuffle=True
                ).history
        else:
            with device(device_name):
                X_train, Y_train = self.prepare_dataset(TRAINING_SET_SIZE_PER_EPOCH)
                X_val, Y_val = self.prepare_dataset(VALIDATION_SET_SIZE_PER_EPOCH)

                history_training_info = self.architecture.fit(
                    X_train, Y_train,
                    epochs=real_epochs,
                    callbacks=callbacks,
                    validation_data=[X_val, Y_val],
                    shuffle=True
                ).history

        """
        finished_training_event.set()
        producer.join()
        """

        if self.trained:
            for dict_key in history_training_info:
                self.history_training_info[dict_key] += history_training_info[dict_key]
        else:
            self.history_training_info = history_training_info
            self.trained = True

    def plot_bias(self):
        raise NotImplementedError

    def plot_predicted_and_ground_truth_distribution(self):
        raise NotImplementedError

    def plot_predicted_and_ground_truth_histogram(self):
        raise NotImplementedError

    def micro_f1_score(self, trajectories=None):
        if trajectories is None:
            trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)
        
        ground_truth = np.argmax(self.transform_trajectories_to_output(trajectories), axis=-1)
        Y_predicted = self.predict(trajectories)
        return f1_score(ground_truth, Y_predicted, average="micro")

    def mae_score(self, trajectories=None):
        if trajectories is None:
            trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)

        ground_truth = self.transform_trajectories_to_output(trajectories)
        Y_predicted = self.predict(trajectories).flatten()

        return mean_absolute_error(ground_truth, Y_predicted)

    def plot_confusion_matrix(self, trajectories=None, normalized=True):
        if trajectories is None:
            trajectories = self.simulator().simulate_trajectories_by_model(VALIDATION_SET_SIZE_PER_EPOCH, self.trajectory_length, self.trajectory_time, self.models_involved_in_predictive_model)
        
        ground_truth = np.argmax(self.transform_trajectories_to_output(trajectories), axis=-1)
        predicted = self.predict(trajectories)

        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=predicted)

        confusion_mat = np.round(confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis], 2) if normalized else confusion_mat

        labels = [model.STRING_LABEL.upper() for model in self.models_involved_in_predictive_model]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        plt.title(f'Confusion Matrix (F1={round(f1_score(ground_truth, predicted, average="micro"),2)})')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        plt.show()

    @property
    def models_involved_in_predictive_model(self):
        return ANDI_MODELS if self.simulator.STRING_LABEL == 'andi' else ALL_MODELS

    def compatible_with_trajectory_for_custom_prediction(self, trajectory):
        return 0.85 * self.trajectory_time <= trajectory.duration <= self.trajectory_time * 1.15
