# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 14.
"""
import collections

import keras.backend as keras
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from keras.layers import Dense, LSTM, Activation, Embedding, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


# AUC for a binary classifier
# noinspection SpellCheckingInspection
def auc(y_true, y_prediction):
    ptas = tf.stack([binary_PTA(y_true, y_prediction, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_prediction, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    bin_sizes = -(pfas[1:] - pfas[:-1])
    s = ptas * bin_sizes
    return keras.sum(s, axis=0)


# PFA, prob False alert for binary classifier
# noinspection PyPep8Naming
def binary_PFA(y_true, y_prediction, threshold=keras.variable(value=0.5)):
    y_prediction = keras.cast(y_prediction >= threshold, 'float32')
    # N = total number of negative labels
    N = keras.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = keras.sum(y_prediction - y_prediction * y_true)
    return FP / N


# PTA, prob True alert for binary classifier
# noinspection PyPep8Naming
def binary_PTA(y_true, y_prediction, threshold=keras.variable(value=0.5)):
    y_prediction = keras.cast(y_prediction >= threshold, 'float32')
    # P = total number of positive labels
    P = keras.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = keras.sum(y_prediction * y_true)
    return TP / P


def count_sentences(sentences):
    max_sentence_length = 0  # the maximum number of words in a sentence
    word_frequents = collections.Counter()  # frequency for each word
    sentence_number = len(sentences)  # total number of records
    word_number = 0  # total number of words

    for words in sentences:
        if len(words) > max_sentence_length:
            max_sentence_length = len(words)
        for word in words:
            word_frequents[word] += 1
        word_number += len(words)

    vocabulary_size = len(word_frequents)

    print(
        'max_sentence_length: {}, vocabulary_size: {}, sentence_number: {}, word_number: {}'.format(max_sentence_length,
                                                                                                    vocabulary_size,
                                                                                                    sentence_number,
                                                                                                    word_number))
    return max_sentence_length, sentence_number, vocabulary_size


def get_embedding_matrix(sentences, embedding_size):
    embed_model = Word2Vec(sentences, sg=0, size=embedding_size, window=5, min_count=1)

    embedding_matrix = embed_model[embed_model.wv.vocab]

    # Index2word / word2index from word2vec output
    index2word = {i: w for i, w in enumerate(embed_model.wv.index2word)}
    word2index = {w: i for i, w in index2word.items()}

    return embedding_matrix, index2word, word2index


def get_x_y(sentences, targets, max_sentence_length, num_recs, word2index):
    # Construct input and outputs for keras
    x = np.empty((num_recs,), dtype=list)
    y = targets
    i = 0

    for words in sentences:
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index['UNK'])
        x[i] = seqs
        i += 1
    x = sequence.pad_sequences(x, maxlen=max_sentence_length)

    return x, y


def get_model(flags, vocabulary_size, max_sentence_length, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocabulary_size, flags.embedding_size,
                        input_length=max_sentence_length,
                        weights=[embedding_matrix],
                        mask_zero=True, trainable=False))
    if flags.dropout is None:
        model.add(LSTM(flags.hidden_unit))
    else:
        model.add(Dropout(flags.dropout))
        model.add(LSTM(flags.hidden_unit))
    model.add(Dense(1, activation='sigmoid'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

    print(model.summary())

    return model


def to_excel(dataframe, dir, file_name):
    writer = pd.ExcelWriter(dir + file_name + '.xlsx')
    dataframe.to_excel(writer)


def evaluate(flags, model, x_test, y_test, max_sentence_length, index2word):
    # Evaluate the model¶
    loss_test, acc_test, auc_test = model.evaluate(x_test, y_test, batch_size=flags.batch_size)

    test_result = []
    for x, y in zip(x_test, y_test):
        x = x.reshape(1, max_sentence_length)
        y_prediction = model.predict(x)[0][0]
        sentence = " ".join([index2word[word] for word in x[0].tolist() if word != 0])
        test_result.append({
            'prediction': y_prediction,
            'label': y,
            'sentence': sentence,
        })

    test_result = pd.DataFrame(test_result)

    return test_result, loss_test, acc_test, auc_test


def run_training(flags, sentences, targets):
    """

    :param flags:
    :param sentences: (List[List[str]]) A list of words. Words is a list of word.
    :param targets: (List[int]) A list of target, 0 or 1. 0 means a negative profit and 1 means a positive profit.
    """
    max_sentence_length, sentence_number, vocabulary_size = count_sentences(sentences)
    embedding_matrix, index2word, word2index = get_embedding_matrix(sentences, flags.embedding_size)
    x, y = get_x_y(sentences, targets, max_sentence_length, sentence_number, word2index)

    # Split whole data into train set and test set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=flags.test_rate)

    # Define a model.
    model = get_model(flags, vocabulary_size, max_sentence_length, embedding_matrix)

    # Teach the model.
    model.fit(x_train, y_train, batch_size=flags.batch_size, epochs=flags.num_epochs,
              validation_data=(x_test, y_test))

    # Evaluate the model.
    test_result, loss_test, acc_test, auc_test = evaluate(flags, model, x_test, y_test, max_sentence_length, index2word)

    # Save the test result as an excel file.
    to_excel(test_result, flags.excel_dir,
             'test_result_{}_{}_{}_{}_{}'.format(flags.embedding_size, flags.batch_size, flags.num_epochs,
                                                 flags.dropout,
                                                 flags.hidden_unit))

    print("Test loss: %.3f, accuracy: %.3f, auc: %.3f" % (loss_test, acc_test, auc_test))
