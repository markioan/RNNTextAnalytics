import os
import re
import pandas as pd
import numpy as np
import nltk
import pickle

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from gensim.models.wrappers import FastText

from definitions import DATA_DIR

import logging

logger = logging.getLogger(__name__)

DATASET_FILENAME = 'stack-overflow-data.csv'
DATASET_PICKLE_FILENAME_TF = 'stack-overflow-pickle-tf'
DATASET_PICKLE_FILENAME_FTC = 'stack-overflow-pickle-ftc'
EMBEDDINGS_FILENAME = 'cc.en.300.bin'
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


def preprocess_full_dataset(df, input_ins='as_tf_idf'):
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

    # Extra preprocesing for the TF - IDF
    if input_ins == 'as_tf_idf':
        # Apply Stemmer
        stemmer = PorterStemmer()
        lambda_exrp = lambda text: ' '.join([stemmer.stem(word.strip()) for word in text.split()])
        df['post'] = df['post'].apply(lambda_exrp)

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


def load_embeddings(input_data, text_field, minimized=False):
    """
    Reads and Loads the embeddings. If partial is enabled then it returns only the embedding for
    the words that is only used into the given <input_data>.
    :param input_data (pandas.DataFrame): The dataframe of the loaded Dataset,
    :param text_field' (str): The key name of the column that the test is contained,
    :param minimized (bool): If true, then load the minimized version of the embeddings,

    :return (dict or Word2Vec): It returns a dictionary with the minified version of the embeddings
                                of the vec or the whole embeddings object (Word2Vec).

    """

    def load_embeddings():
        embeddings_path = os.path.join(DATA_DIR, EMBEDDINGS_FILENAME)
        return FastText.load_fasttext_format(embeddings_path)

    if minimized:
        try:
            minimized_path = os.path.join(DATA_DIR, MINIMIZED_EMBEDDINGS_FILENAME)
            with open(minimized_path, 'rb') as minimized_pickle:
                embeddings = pickle.load(minimized_pickle)

        except Exception as e:
            logger.exception(e)
            embeddings = minimize_embeddings(input_data, load_embeddings(), text_field)

    else:
        embeddings = load_embeddings()

    return embeddings


def load_dataset(tags_categories='__all__', load_from_pickle=True, input_ins='as_tf_idf'):
    """
    Loads and returns the Dataset as a DataFrame. The returned Dataframe will be proccesed.

    :param tags_categories: If '__all__' is given then all the dataset is parsed.
                    Otherwise a list of the class names should be passed that
                    will filter the dataset against.
    :load_from_pickle (bool): If True then tries to load from pickle file, Otherwise it
                              loads the initial dataset.
    :input_ins (str): If 'as_tf_idf' is given then it generates tf-idf vectors per text row
                      If 'as_centroids' is given then it generates centroids per text row
    :return: A DataFrame filled with the whole or a subset of the dataset loaded.

    """

    def load_dataset_and_preprocess():
        dataset_path = os.path.join(DATA_DIR, DATASET_FILENAME)
        dataset_df = pd.read_csv(dataset_path)
        return preprocess_full_dataset(dataset_df, input_ins)

    assert tags_categories == '__all__' or isinstance(tags_categories, list)  or isinstance(tags_categories, tuple), \
        ("Argument <tags_categories> should be a type of 'list' or 'tuple' or a string with explicit value '__all__'."
         "Instead it got the value {}".format(tags_categories))

    if input_ins == 'as_tf_idf':
        pickle_path = os.path.join(DATA_DIR, DATASET_PICKLE_FILENAME_TF)
    else:
        pickle_path = os.path.join(DATA_DIR, DATASET_PICKLE_FILENAME_FTC)

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



def preprocess_data(input_data, label_field, text_field, input_ins='as_tf_idf', embeddings=None, **kwargs):
    """
    Generates the train-test data for the model based on the given arguments.

    Args:
        'input_data' (pandas.DataFrame): The dataset Dataframe that will be splitted in train-test data.

        'label_field' (str): The key name of the column that the dataset's classes are contained,

        'text_field' (str): The key name of the column that the test is contained,

        'embeddings' (dict or Word2Vec): It is required when precessing as 'as_centroids'.
                                         Represents either a key:value pair of word: word2vec vectors representation,
                                         or the whole Word2Vec instance model

        'input_ins' (str): If 'as_tf_idf' is given then it generates tf-idf vectors per text row
                     If 'as_centroids' is given then it generates centroids per text row
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
    standardize = kwargs.get('standardize', False)


    full_train, test = train_test_split(input_data,
                                        test_size=cv_split_full,
                                        random_state=1596,
                                        stratify=input_data[label_field])
    train, train_dev = train_test_split(full_train,
                                        test_size=cv_split_dev,
                                        random_state=1596,
                                        stratify=full_train[label_field])

    if input_ins == 'as_tf_idf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                     max_features=5000,
                                     sublinear_tf=True).fit(train[text_field])
        x_train = vectorizer.transform(train[text_field]).toarray()
        x_train_dev = vectorizer.transform(train_dev[text_field]).toarray()
        x_test = vectorizer.transform(test[text_field]).toarray()

    else:
        x_train = np.array(list(map(lambda text: text_centroid(text, embeddings), train[text_field])))
        x_train = np.stack(x_train, axis=0)

        x_train_dev = np.array(list(map(lambda text: text_centroid(text, embeddings), train_dev[text_field])))
        x_train_dev = np.stack(x_train_dev, axis=0)

        x_test = np.array(list(map(lambda text: text_centroid(text, embeddings), test[text_field])))
        x_test = np.stack(x_test, axis=0)

    # Extra preprocessing in transformed data
    if standardize:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_train_dev = scaler.transform(x_train_dev)
        x_test = scaler.transform(x_test)


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
        'y_test': y_test
    }


if __name__ == "__main__":
    data = load_dataset()
    print(data['tags'].categories)
