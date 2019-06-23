from collections import Iterable

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras_tqdm import TQDMNotebookCallback
from talos import Reporting
from talos.model import early_stopper

from app.metrics import f1


def load_model(x_train, y_train, x_train_dev, y_train_dev, kwargs):
    """
    The callback method that will be used from the Talos API in order to
    generate a configurable Keras  MLP model.
    :return (tuple): A tuple with the history object of the model train and the ,
                     generated Keras model itself.
    """

    def generate_hidden_layers(model_nn, n_labels, **kwargs):
        """
        In order to create hidden layer for the constructed model, the kwargs
        parameters must contain the keys:
        1) 'number_of_hidden_layers' : Describes the number of the layers that will be generated
        2) 'first_neuron': Describes the number of nodes of the model's first layer
        3) 'dropout':  Describe the portion of the set that will be dropouted after each Layer
        The minimum value of nodes that can be applied to a hidden layer is n_classses.

        """
        hidden_layers = kwargs.get('number_of_hidden_layers')
        if hidden_layers:
            for h_layer in range(1, hidden_layers + 1):
                nodes = kwargs.get('first_neuron') / (2 * h_layer)
                nodes = int(nodes if nodes > n_labels else n_labels)
                model_nn.add(Dense(nodes, activation=kwargs.get('activation', 'relu')))
                model_nn.add(Dropout(kwargs.get('dropout', 0.5)))
        return  model_nn

    n_labels = y_train.shape[1]
    visualize_process = kwargs.get('visualize_process', False)
    with_early_stoping = kwargs.get('early_stopping', True)

    model = Sequential()
    model.add(Dense(units=kwargs.get('first_neuron', 2),
                    input_dim=x_train.shape[1],
                    activation=kwargs.get('activation', 'relu')))

    # Dropout probability in order to avoid overfitting.
    model.add(Dropout(kwargs.get('dropout', 0.5)))

    # Apply hidden layers
    model = generate_hidden_layers(model, n_labels, **kwargs)

    # last Hidden layer
    # Mutual exclusive Classes
    model.add(Dense(n_labels, activation='softmax'))

    # Apply default Callback methods
    if with_early_stoping:
        stopper = EarlyStopping(monitor=kwargs.get('early_stopping_config__monitor', 'val_f1'),
                                min_delta=kwargs.get('early_stopping_config__min_delta', 0),
                                patience=kwargs.get('early_stopping_config__patience', 5),
                                mode=kwargs.get('early_stopping_config__mode', 'auto'),

                                )
        default_callbacks = [stopper]
    else:
        default_callbacks = []

    # Apply extra monitoring Callback methods
    if visualize_process:
        print(model.summary())
        checkpoint = ModelCheckpoint(kwargs.get('model_type', 'keras_tf_idf_model'),
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')

        checkpoint2 = ModelCheckpoint(kwargs.get('model_type', 'keras_tf_idf_model'),
                                     monitor='val_f1',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')

        extra_callbacks = [checkpoint, TQDMNotebookCallback(), checkpoint2]
    else:
        extra_callbacks = []

    # Model compilation parameterized with
    model.compile(loss='categorical_crossentropy',
                  optimizer=kwargs.get('optimizer', 'Adam'),
                  metrics=[f1, 'categorical_accuracy'])

    history = model.fit(x_train,y_train,
                        batch_size=kwargs.get('batch_size', 32),
                        epochs=kwargs.get('epochs', 10),
                        verbose=0,
                        callbacks=default_callbacks + extra_callbacks,
                        validation_data=(x_train_dev, y_train_dev),
                        shuffle=True)

    return history, model


def find_best_model_over_scan_logs(metric_weight='val_f1', *filepaths):
    """
    Finds the best model against multiple Talos scanned configurations.
    The scan configuration that has the maximum <metric_weight> is
    the one that is described as the best.

    :param metric_weight: It describes the evaluation metric that will be user
                          as a qualifier for the best model.
    :param filepaths: An iterable that will contains the correct filepaths of the
                      saved Talos configurations

    :return (dict): A dictionary with the best model configuration.
    """
    assert metric_weight is not None, "Argument <metric_weight> can not be None."
    assert isinstance(filepaths, Iterable), "Argument <filepaths> must be iterable "

    # Cre
    config_pd = pd.concat(map(lambda file: Reporting(file).data, filepaths))
    config_pd.index = range(config_pd.shape[0])

    best_model_idx = config_pd[metric_weight].idxmax()
    best_model = config_pd.loc[best_model_idx].to_dict()

    for key, value in best_model.items():
        if isinstance(value, float) and value >= 1:
            best_model[key] = int(value)
        elif value == 'False' or value == 'True':
            best_model[key] = value == 'True'
    return best_model