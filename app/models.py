import logging
import os
import pickle

from collections import Iterable

import pandas as pd
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, Dense, Dropout, Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras_tqdm import TQDMNotebookCallback
from talos import Reporting

from app.layers import DeepAttention
from app.metrics import *
from definitions import MAX_SEQUENCE_LENGTH, MAX_WORDS, DATA_DIR

logger = logging.getLogger(__name__)

def load_bi_gru_model(x_train, y_train, x_train_dev, y_train_dev, params):
    """
    The callback method that will be used from the Talos API in order to
    generate a configurable Keras RNN Model with Bidirection GRUs.
    :return (tuple): A tuple with the history object of the model train and the ,
                     generated Keras model itself.
    """
    inputs = Input((MAX_SEQUENCE_LENGTH,))
    EMBEDDING_DIM = params.get('embedding_dim')
    GRU_SIZE = params.get('gru_size', 200)
    DENSE = params.get('dense', 300)
    N_CLASSES = y_train.shape[1]
    embeddings_matrix_path = os.path.join(DATA_DIR, params['embeddings_matrix_path'])
    with open(embeddings_matrix_path, 'rb') as embeddings_matrix_pickle:
        embeddings_matrix = pickle.load(embeddings_matrix_pickle)
    visualize_process = params.get('visualize_process', False)
    visualize_process = (visualize_process if isinstance(visualize_process, bool)
                         else visualize_process == 'True')
    with_early_stoping = params.get('with_early_stoping', True)
    with_early_stoping = (with_early_stoping if isinstance(with_early_stoping, bool)
                          else with_early_stoping == 'True')
    multistack_run = params.get('multistack_run', False)
    multistack_run = (multistack_run if isinstance(multistack_run, bool)
                      else multistack_run == 'True')

    # Apply default Callback methods
    if with_early_stoping:
        stopper = EarlyStopping(monitor=params.get('early_stopping_config__monitor', 'val_f1'),
                                min_delta=params.get('early_stopping_config__min_delta', 0),
                                patience=params.get('early_stopping_config__patience', 5),
                                mode=params.get('early_stopping_config__mode', 'auto'),

                                )
        default_callbacks = [stopper]
    else:
        default_callbacks = []

    # Apply extra monitoring Callback methods
    if visualize_process:
        checkpoint = ModelCheckpoint(params['model_type'],
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')

        checkpoint2 = ModelCheckpoint(params['model_type'],
                                      monitor='val_f1',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='max')

        extra_callbacks = [checkpoint, TQDMNotebookCallback(), checkpoint2]
    else:
        extra_callbacks = []

    # Add an embedding layer with 0.2 dropout probability
    embeddings = Embedding(MAX_WORDS + 2, EMBEDDING_DIM, weights=[embeddings_matrix],
                           input_length=MAX_SEQUENCE_LENGTH, mask_zero=True, trainable=False)(inputs)
    drop_emb = Dropout(params['embeddings_dropout'])(embeddings)

    # add a bidirectional gru layer with variational (recurrent) dropout
    bi_gru = Bidirectional(GRU(GRU_SIZE,
                               return_sequences=True,
                               recurrent_dropout=params['var_dropout']))(drop_emb)
    if multistack_run:
        # In Case of multistack  RNN
        deep_attention_entry = GRU(GRU_SIZE,
                                   return_sequences=True,
                                   recurrent_dropout=params['var_dropout'])(bi_gru)
    else:
        deep_attention_entry = bi_gru

    # add a deep self attention layer
    x, attn = DeepAttention(return_attention=True)(deep_attention_entry)

    # add a hidden MLP layer
    drop = Dropout(params["mlp_dropout"])(x)
    out = Dense(DENSE, activation=params['rnn_activation'])(x)

    # add the output MLP layer
    out = Dense(N_CLASSES, activation=params['mlp_activation'])(out)
    model = Model(inputs, out)

    if visualize_process:
        print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=[precision, recall, f1, accuracy, 'categorical_accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=params.get('batch_size', 32),
                        epochs=params.get('epochs', 10),
                        verbose=0,
                        callbacks=default_callbacks + extra_callbacks,
                        validation_data=(x_train_dev, y_train_dev),
                        shuffle=True)

    return history, model


def load_bi_lstm_model(x_train, y_train, x_train_dev, y_train_dev, params):
    """
    The callback method that will be used from the Talos API in order to
    generate a configurable Keras RNN Model with Bidirection GRUs.
    :return (tuple): A tuple with the history object of the model train and the ,
                     generated Keras model itself.
    """
    inputs = Input((MAX_SEQUENCE_LENGTH,))
    EMBEDDING_DIM = params.get('embedding_dim')
    LSTM_SIZE = params.get('lstm_size', 200)
    DENSE = params.get('dense', 300)
    N_CLASSES = y_train.shape[1]
    embeddings_matrix_path = os.path.join(DATA_DIR, params['embeddings_matrix_path'])
    with open(embeddings_matrix_path, 'rb') as embeddings_matrix_pickle:
        embeddings_matrix = pickle.load(embeddings_matrix_pickle)

    visualize_process = params.get('visualize_process', False)
    visualize_process = (visualize_process if isinstance(visualize_process, bool)
                         else visualize_process == 'True')
    with_early_stoping = params.get('with_early_stoping', True)
    with_early_stoping = (with_early_stoping if isinstance(with_early_stoping, bool)
                          else with_early_stoping == 'True')
    multistack_run = params.get('multistack_run', False)
    multistack_run = (multistack_run if isinstance(multistack_run, bool)
                      else multistack_run == 'True')

    # Apply default Callback methods
    if with_early_stoping:
        stopper = EarlyStopping(monitor=params.get('early_stopping_config__monitor', 'val_f1'),
                                min_delta=params.get('early_stopping_config__min_delta', 0),
                                patience=params.get('early_stopping_config__patience', 5),
                                mode=params.get('early_stopping_config__mode', 'auto'),

                                )
        default_callbacks = [stopper]
    else:
        default_callbacks = []

    # Apply extra monitoring Callback methods
    if visualize_process:
        checkpoint = ModelCheckpoint(params['model_type'],
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')

        checkpoint2 = ModelCheckpoint(params['model_type'],
                                      monitor='val_f1',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='max')

        extra_callbacks = [checkpoint, TQDMNotebookCallback(), checkpoint2]
    else:
        extra_callbacks = []

    # Add an embedding layer with 0.2 dropout probability
    embeddings = Embedding(MAX_WORDS+2, EMBEDDING_DIM, weights=[embeddings_matrix],
                           input_length=MAX_SEQUENCE_LENGTH, mask_zero=True, trainable=False)(inputs)
    drop_emb = Dropout(params['embeddings_dropout'])(embeddings)

    # add a bidirectional gru layer with variational (recurrent) dropout
    bi_lstm = Bidirectional(LSTM(LSTM_SIZE,
                                 return_sequences=True,
                                 recurrent_dropout=params['var_dropout']))(drop_emb)
    if multistack_run:
        # In Case of multistack  RNN
        deep_attention_entry = LSTM(LSTM_SIZE,
                                    return_sequences=True,
                                    recurrent_dropout=params['var_dropout'])(bi_lstm)
    else:
        deep_attention_entry = bi_lstm

    #add a deep self attention layer
    x, attn = DeepAttention(return_attention=True)(deep_attention_entry)

    # add a hidden MLP layer
    drop = Dropout(params["mlp_dropout"])(x)
    out = Dense(DENSE, activation=params['rnn_activation'])(x)

    # add the output MLP layer
    out = Dense(N_CLASSES, activation=params['mlp_activation'])(out)
    model = Model(inputs, out)

    if visualize_process:
        print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=[precision, recall, f1, accuracy, 'categorical_accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=params.get('batch_size', 32),
                        epochs=params.get('epochs', 10),
                        verbose = 0,
                        callbacks= default_callbacks + extra_callbacks,
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

    talos_configs = []
    for file in filepaths:
        try:
            talos_configs.append(Reporting(file).data)
        except Exception as e:
            logger.warning(e)

    config_pd = pd.concat(talos_configs)
    # print(config_pd)
    config_pd.index = range(config_pd.shape[0])

    best_model_idx = config_pd[metric_weight].idxmax()
    best_model = config_pd.loc[best_model_idx].to_dict()

    for key, value in best_model.items():
        if isinstance(value, float) and value >= 1:
            best_model[key] = int(value)
        elif value == 'False' or value == 'True':
            best_model[key] = value == 'True'
    return best_model