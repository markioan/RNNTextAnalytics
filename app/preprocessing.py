import os
import re
import pandas as pd
import numpy as np
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from definitions import DATA_DIR

from definitions import MAX_WORDS, MAX_SEQUENCE_LENGTH

import logging

logger = logging.getLogger(__name__)

DATASET_FILENAME = 'stack-overflow-data.csv'
DATASET_PICKLE_FILENAME = 'stack-overflow-pickle'
EMBEDDINGS_FILENAME = 'cc.en.300.vec'
EMBEDDINGS_VOC = 'fasttext_voc'
EMBEDDINGS_VEC = 'fasttext.npy'
EMBEDDINGS_MATRIX_PICKLE_FILENAME = 'embeddings-matrix-pickle'
MINIMIZED_EMBEDDINGS_FILENAME = 'minimized_embeddings'

# Data Processing methods

def text_centroid(text, model):
    text_vec = []
    counter = 0
    sent_text = nltk.sent_tokenize(text)
    for sentence in sent_text:
        sent_tokenized = nltk.word_tokenize(sentence)
        for word in sent_tokenized:
            try:
                if counter == 0:
                    text_vec = model[word.lower()]
                else:
                    text_vec = np.add(text_vec, model[word.lower()])
                counter += 1
            except:
                pass

    return np.asarray(text_vec) / counter


def preprocess_full_dataset(df):
    """
    The data preprocessing of the full dataset. The only extra preprocessing that is implemented
    to the dataset in case of the *tf_idf* model is the *stemming* of each word
    :param df (pandas.DataFrame):  The dataframe of the loaded Dataset
    :param input_ins:
    :return:
    """
    df['tags'] = pd.Categorical(df.tags)

    # convert text to lowercase
    df['post'] = df['post'].str.lower()

    # remove numbers
    df['post'] = df['post'].str.replace("[0-9]", " ")

    # # # remove stopwords
    stop_words = stopwords.words('english')
    df['post'] = (df['post'].apply(lambda text: ' '.join([word.strip() for word in text.split()
                                                          if word not in stop_words])))

    # remove links
    links_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['post'] = df['post'].apply(lambda text: re.sub(links_regex, "", text))

    # remove punctuation characters
    punc_reqex = '[!,.:;-](?= |$)'
    df['post'] = df['post'].apply(lambda text: re.sub(punc_reqex, r'', text))

    return df


# Loading & Saving Data Methods
def minimize_embeddings(input_data, emb_model, text_field, save=True):
    """
    Finds the embeddings of the words that exist in the <input_data>. If <save> is enabled
    then it stores it in a pickle file.

    :param input_data (pandas.DataFrame): The dataframe of the loaded Dataset,
    :param emb_model (Word2Vec): the loaded vocabulary's embeddings.
    :param text_field (str): The key name of the column that the test is contained,
    :param save (bool):
    :return (dic):

    """
    distinct_words = []
    for text in input_data[text_field]:
        sent_text = nltk.sent_tokenize(text)
        for sentence in sent_text:
            sent_tokenized = nltk.word_tokenize(sentence)
            distinct_words += list(set(sent_tokenized))
    distinct_words = list(set(distinct_words))
    minimized = {word: emb_model[word] for word in distinct_words if word in emb_model}
    if save:
        minimized_path = os.path.join(DATA_DIR, MINIMIZED_EMBEDDINGS_FILENAME)
        with open(minimized_path, 'wb') as minimized_pickle:
            pickle.dump(minimized, minimized_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    return minimized


def load_embeddings(load_from_pickle=True):
    """
    Reads and Loads the embeddings. If partial is enabled then it returns only the embedding for
    the words that is only used into the given <input_data>.
    :param input_data (pandas.DataFrame): The dataframe of the loaded Dataset,
    :param text_field' (str): The key name of the column that the test is contained,
    :param minimized (bool): If true, then load the minimized version of the embeddings,

    :return (dict or Word2Vec): It returns a dictionary with the minified version of the embeddings
                                of the vec or the whole embeddings object (Word2Vec).

    """

    def open_embeddings_vocabulary():
        idx = 0
        vocab = {}
        vocab_path = os.path.join(DATA_DIR, EMBEDDINGS_FILENAME)
        with open(vocab_path, 'r', encoding="utf-8", newline='\n', errors='ignore') as f:
            for l in f:
                line = l.rstrip().split(' ')
                if idx == 0:
                    vocab_size = int(line[0]) + 2
                    dim = int(line[1])
                    vecs = np.zeros(vocab_size * dim).reshape(vocab_size, dim)
                    vocab["__PADDING__"] = 0
                    vocab["__UNK__"] = 1
                    idx = 2
                else:
                    vocab[line[0]] = idx
                    emb = np.array(line[1:]).astype(np.float)
                    if (emb.shape[0] == dim):
                        vecs[idx, :] = emb
                        idx += 1
                    else:
                        continue
        return vocab, vecs

    embeddings_voc_path = os.path.join(DATA_DIR, EMBEDDINGS_VOC)
    embeddings_vec_path = os.path.join(DATA_DIR, EMBEDDINGS_VEC)

    if load_from_pickle:
        try:
            with open(embeddings_voc_path, 'rb') as embeddings_voc_pickle:
                embeddings_voc = pickle.load(embeddings_voc_pickle)

            with open(embeddings_vec_path, "rb") as embeddings_vec_np:
                embeddings_vec = np.load(embeddings_vec_np)

        except Exception as e:
            logger.exception(e)
            embeddings_voc, embeddings_vec  = open_embeddings_vocabulary()
            pickle.dump(embeddings_voc, open(embeddings_voc_path, 'wb'))
            np.save(embeddings_vec_path, embeddings_vec)

    else:
        embeddings_voc, embeddings_vec  = open_embeddings_vocabulary()

    return embeddings_voc, embeddings_vec


def load_dataset(tags_categories='__all__', load_from_pickle=True):
    """
    Loads and returns the Dataset as a DataFrame. The returned Dataframe will be proccesed.

    :param tags_categories: If '__all__' is given then all the dataset is parsed.
                    Otherwise a list of the class names should be passed that
                    will filter the dataset against.
    :load_from_pickle (bool): If True then tries to load from pickle file, Otherwise it
                              loads the initial dataset.
    :return: A DataFrame filled with the whole or a subset of the dataset loaded.

    """

    def load_dataset_and_preprocess():
        dataset_path = os.path.join(DATA_DIR, DATASET_FILENAME)
        dataset_df = pd.read_csv(dataset_path)
        return preprocess_full_dataset(dataset_df)

    assert tags_categories == '__all__' or isinstance(tags_categories, list)  or isinstance(tags_categories, tuple), \
        ("Argument <tags_categories> should be a type of 'list' or 'tuple' or a string with explicit value '__all__'."
         "Instead it got the value {}".format(tags_categories))

    pickle_path = os.path.join(DATA_DIR, DATASET_PICKLE_FILENAME)

    if load_from_pickle:
        try:
            dataset_df = pd.read_pickle(pickle_path)
        except Exception as e:
            logger.warning(e)
            dataset_df = load_dataset_and_preprocess()
            dataset_df.to_pickle(pickle_path)
    else:
        dataset_df = load_dataset_and_preprocess()
        dataset_df.to_pickle(pickle_path)

    return dataset_df if tags_categories == '__all__' else dataset_df.loc[dataset_df['tags'].isin(tags_categories)]


def preprocess_data(input_data, label_field, text_field, **kwargs):
    """
    Generates the train-test data for the model based on the given arguments.

    Args:
        'input_data' (pandas.DataFrame): The dataset Dataframe that will be splitted in train-test data.

        'label_field' (str): The key name of the column that the dataset's classes are contained,

        'text_field' (str): The key name of the column that the test is contained,

    It returns a dictionary with the below structure:
        {
            'x_train' (numpy.array),
            'x_test' (numpy.array),
            'y_train' (numpy.array),
            'y_test' (numpy.array),
        }

    """
    assert all([label_field, text_field]), \
        "Fields <label_field>, <text_field> cannot be None or empty"

    cv_split_full = kwargs.get('cv_split_full', 0.3)
    cv_split_dev = kwargs.get('cv_split_dev', 0.2)


    full_train, test = train_test_split(input_data,
                                        test_size=cv_split_full,
                                        random_state=1596,
                                        stratify=input_data[label_field])
    train, train_dev = train_test_split(full_train,
                                        test_size=cv_split_dev,
                                        random_state=1596,
                                        stratify=full_train[label_field])


    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='__UNK__')
    tokenizer.fit_on_texts(train[text_field])
    train_seqs = tokenizer.texts_to_sequences(train[text_field])
    dev_seqs = tokenizer.texts_to_sequences(train_dev[text_field])
    test_seqs = tokenizer.texts_to_sequences(test[text_field])
    x_train = pad_sequences(train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    x_train_dev = pad_sequences(dev_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    x_test = pad_sequences(test_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    mlb = MultiLabelBinarizer()

    y_train = mlb.fit_transform(train[[label_field]].values.tolist())

    y_train_dev = mlb.transform(train_dev[[label_field]].values.tolist())

    y_test = mlb.transform(test[[label_field]].values.tolist())

    return {
        'x_train': x_train,
        'x_train_dev': x_train_dev,
        'x_test': x_test,
        'y_train': y_train,
        'y_train_dev': y_train_dev,
        'y_test': y_test,
        'words_index': tokenizer.word_index
    }


def save_embeddings_matrix(embeddings_voc, embeddings_vec,  words_index, filename=EMBEDDINGS_MATRIX_PICKLE_FILENAME):
    embeddings_dim = embeddings_vec.shape[1]
    # Extra values of '__UNK__' and '__PADDING__'
    embedding_matrix = np.zeros((MAX_WORDS + 2, embeddings_dim))
    for word, i in words_index.items():
        if i > MAX_WORDS:
            continue
        try:
            embedding_vector = embeddings_vec[embeddings_voc[word], :]
            embedding_matrix[i] = embedding_vector
        except:
            pass

    embeddings_matrix_pickle_path = os.path.join(DATA_DIR, filename)
    pickle.dump(embedding_matrix,  open(embeddings_matrix_pickle_path, 'wb'))
    return filename


if __name__ == "__main__":
    data = load_dataset()
    print(data['tags'].categories)
