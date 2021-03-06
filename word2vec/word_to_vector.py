# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 14.
"""
import collections
from pathlib import Path

import keras.backend as keras
import numpy as np
import pandas as pd
import progressbar
import tensorflow as tf
from gensim.models import Word2Vec
from keras import metrics
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.models import Sequential, save_model
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
                        mask_zero=True))
    if flags.dropout is None:
        model.add(LSTM(flags.hidden_unit))
    else:
        model.add(Dropout(flags.dropout))
        model.add(LSTM(flags.hidden_unit))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=[metrics.binary_accuracy, auc])

    print(model.summary())

    return model


def to_excel(dataframe, dir, file_name):
    writer = pd.ExcelWriter(dir + file_name + '.xlsx')
    dataframe.to_excel(writer)


# noinspection PyPep8Naming
def to_html(test_result, file_name, flags, company_name):
    y = test_result['label']
    predict = np.round(test_result['prediction'])
    TP = sum(y * predict)  # 실제 1을 1이라고 예측한것
    TN = sum((y - 1) * (predict - 1))  # 실제 0을 0이라고 예측한것
    FP = -1 * sum((y - 1) * predict)  # 실제 0을 1이라고 예측한것
    FN = -1 * sum(y * (predict - 1))  # 실제 1을 0이라고 예측한것
    Precision = TP / (TP + FP + 1e-20)  # 1이라고 예측한것 중 실제 1인것의 비중
    Recall = TP / (TP + FN + 1e-20)  # 실제 1인것들 중에서 예측결과가 1인 것의 비중
    Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 정확히 예측(즉, 1을 1이라고, 0을 0이라고 예측)한 것의 비중
    F1Score = 2 / (1 / Precision + 1 / Recall + 1e-20)  # harmonic mean

    f = open(flags.html_dir + file_name + '.html', 'w')

    html_header = '<!DOCTYPE html>' \
                  '<meta charset="utf-8">' \
                  '<html><body>' \
                  '<h3>company_name={},<br>' \
                  'embedding_size={},<br>' \
                  'batch_size={},<br>' \
                  'num_epochs={},<br>' \
                  'dropout={},<br>' \
                  'hidden_unit={}</h3>'.format(company_name,
                                               flags.embedding_size,
                                               flags.batch_size,
                                               flags.num_epochs,
                                               flags.dropout,
                                               flags.hidden_unit)
    html_table = """
        <style type="text/css">
        .tg  {border-collapse:collapse;border-spacing:0;}
        .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
        .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
        .tg .tg-s6z2{text-align:center}
        .tg .tg-baqh{text-align:center;vertical-align:top}
        .tg .tg-804w{font-family:Arial, Helvetica, sans-serif !important;;text-align:center;vertical-align:top}
        .tg .tg-431l{font-family:Arial, Helvetica, sans-serif !important;;text-align:center}
        .tg .tg-szxb{font-family:"Arial Black", Gadget, sans-serif !important;;text-align:center}
        </style>
        <table class="tg">
          <tr>
            <th class="tg-s6z2" colspan="2" rowspan="1">F1 score</th>
            <th class="tg-431l" colspan="2">Prediction</th>
            <th class="tg-804w" rowspan="3">Recall</th>
          </tr>
          <tr>
            <td class="tg-baqh" colspan="2">%.4f</td>
            <td class="tg-szxb">0</td>
            <td class="tg-szxb">1</td>
          </tr>
          <tr>
            <td class="tg-431l" rowspan="2">Label</td>
            <td class="tg-szxb">0</td>
            <td class="tg-s6z2">%d</td>
            <td class="tg-s6z2">%d</td>
          </tr>
          <tr>
            <td class="tg-szxb">1</td>
            <td class="tg-s6z2">%d</td>
            <td class="tg-s6z2">%d</td>
            <td class="tg-baqh">%.4f</td>
          </tr>
          <tr>
            <td class="tg-804w" colspan="3">Precision</td>
            <td class="tg-baqh">%.4f</td>
            <td class="tg-baqh">%.4f</td>
        </table>
    """ % (F1Score, TN, FP, FN, TP, Recall, Precision, Accuracy)
    html_footer = '</body></html>'

    f.write(html_header)
    f.write(html_table)
    f.write(html_footer)
    f.close()


def evaluate(flags, model, x_test, y_test, index2word):
    # Evaluate the model¶
    test_result = {
        'prediction': [],
        'label': [],
        'sentence': [],
    }

    # Initialize a progressbar.
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=(len(x_test) // flags.batch_size + 1)).start()
    index = 0
    while x_test.size != 0:
        index += 1

        x_test, y_test, x_batch, y_batch = next_test_batch(flags.batch_size, x_test, y_test)

        y_predictions = [prediction[0] for prediction in model.predict(x_batch, batch_size=flags.batch_size)]
        sentences = [" ".join([index2word[word] for word in x if word != 0]) for x in x_batch]

        assert len(y_predictions) == len(x_batch)
        assert len(x_batch) == len(sentences)

        # test_result['prediction'].extend([np.where(y_prediction > 1.0, 1, 0) for y_prediction in y_predictions])
        test_result['prediction'].extend(y_predictions)
        test_result['label'].extend(y_batch)
        test_result['sentence'].extend(sentences)

        # Update the progressbar.
        bar.update(index)

    # Finish the progressbar.
    bar.finish()

    test_result = pd.DataFrame(test_result)

    return test_result


def next_test_batch(num, x_test, y_test):
    """
    Return a total of `num` random samples and labels, and delete x_batch and y_batch
    from original x_test and y_test.
    """
    original_len = len(x_test)

    idx = np.arange(0, len(x_test))
    np.random.shuffle(idx)
    idx = idx[:min(num, len(x_test))]
    x_batch = np.asarray([x_test[i] for i in idx])
    y_batch = np.asarray([y_test[i] for i in idx])

    idx.sort()
    idx = idx[::-1]
    for i in idx:
        x_test = np.delete(x_test, i, axis=0)
        y_test = np.delete(y_test, i, axis=0)

    current_len = len(x_test)
    assert original_len == current_len + len(idx)

    return x_test, y_test, x_batch, y_batch


def run_training(company_name, flags, sentences, targets, demonstration=False):
    """

    :param company_name:
    :param flags:
    :param sentences: (List[List[str]]) A list of words. Words is a list of word.
    :param targets: (List[int]) A list of target, 0 or 1. 0 means a negative profit and 1 means a positive profit.
    :param demonstration:
    """

    file_name = 'test_result_{}_{}_{}_{}_{}_{}_{}'.format(company_name,
                                                          flags.learning_rate,
                                                          flags.embedding_size,
                                                          flags.batch_size,
                                                          flags.num_epochs,
                                                          flags.dropout,
                                                          flags.hidden_unit)

    if Path(flags.html_dir + file_name + '.html').exists():
        return

    max_sentence_length, sentence_number, vocabulary_size = count_sentences(sentences)
    embedding_matrix, index2word, word2index = get_embedding_matrix(sentences, flags.embedding_size)
    x, y = get_x_y(sentences, targets, max_sentence_length, sentence_number, word2index)

    # Split whole data into train set and test set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=flags.test_rate)

    # Define a model.
    model = get_model(flags, vocabulary_size, max_sentence_length, embedding_matrix)

    # Teach the model. (x_test, y_test)
    model.fit(x_train, y_train, batch_size=flags.batch_size, epochs=flags.num_epochs,
              validation_data=(x_test, y_test))

    save_model(model, flags.log_dir)

    if demonstration:
        while True:
            try:
                input_sentence = input()
                input_words = input_sentence.split()

                indexed_words = []
                for word in input_words:
                    if word in word2index:
                        indexed_words.append(word2index[word])

                # If indexed_words is longer than max_sentence_length, delete words after max_sentence_length.
                if len(indexed_words) > max_sentence_length:
                    indexed_words = indexed_words[:max_sentence_length]
                else:
                    while len(indexed_words) != max_sentence_length:
                        indexed_words.append(0)

                indexed_words = np.asarray(indexed_words)
                indexed_words = indexed_words.reshape(1, max_sentence_length)
                predict_results = model.predict(indexed_words)
                y_prediction = keras.round(predict_results[0][0])
                if y_prediction == 1:
                    print("Up, {}".format(predict_results))
                else:
                    print("Down, {}".format(predict_results))
            except Exception as e:
                print("Exception: {}".format(e))
    else:
        # Evaluate the model.
        test_result = evaluate(flags, model, x_test, y_test, index2word)

        # Save the test result as an excel file.
        to_excel(test_result, flags.excel_dir, file_name)

        to_html(test_result, file_name, flags, company_name)
