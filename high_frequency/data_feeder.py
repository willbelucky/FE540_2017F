# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 27.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tqdm import tqdm

from data_dealer.data_reader import get_stock_minute_prices, get_stock_master

DATASETS = collections.namedtuple('Datasets', ['train', 'test', 'column_number', 'class_number', 'batch_size'])


class DataSet(object):
    def __init__(self,
                 units,
                 labels,
                 dates,
                 column_number,
                 class_number,
                 batch_size=100,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid unit dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert units.shape[0] == labels.shape[0], (
                    'units.shape: %s labels.shape: %s' % (units.shape, labels.shape))
            self._num_examples = units.shape[0]
        self._seed = seed
        self._dtype = dtype
        self._units = units
        self._labels = labels
        self._dates = dates
        self._batch_size = batch_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.column_number = column_number
        self.class_number = class_number

    @property
    def units(self):
        return self._units

    @property
    def labels(self):
        return self._labels

    @property
    def dates(self):
        return self._dates

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def dtype(self):
        return self._dtype

    @property
    def seed(self):
        return self._seed

    def next_batch(self, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_unit = [1] * self.column_number
            if self.one_hot:
                fake_label = 1
            else:
                fake_label = 0
            fake_date = [i for i in range(self.column_number)]
            return [fake_unit for _ in range(self._batch_size)], [fake_label for _ in range(self._batch_size)], [
                fake_date for _ in range(self._batch_size)]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._units = self.units.iloc[perm0]
            self._labels = self.labels[perm0]
            self._dates = self.dates[perm0]
        # Go to the next epoch
        if start + self._batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            units_rest_part = self._units[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            dates_rest_part = self._dates[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._units = self.units.iloc[perm]
                self._labels = self.labels[perm]
                self._dates = self.dates[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self._batch_size - rest_num_examples
            end = self._index_in_epoch
            units_new_part = self._units[start:end]
            labels_new_part = self._labels[start:end]
            dates_new_part = self._dates[start:end]
            return np.concatenate((units_rest_part, units_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0), np.concatenate((dates_rest_part, dates_new_part), axis=0)
        else:
            self._index_in_epoch += self._batch_size
            end = self._index_in_epoch
            return self._units[start:end], self._labels[start:end], self._dates[start:end]


def to_recurrent_data(data_sets, time_step):
    train_size = len(data_sets.train.labels) - time_step

    units = pd.concat([data_sets.train.units, data_sets.test.units])
    units = dataframe_to_recurrent_ndarray(units, time_step)
    labels = np.concatenate([data_sets.train.labels, data_sets.test.labels])[time_step:]
    dates = np.concatenate([data_sets.train.dates, data_sets.test.dates])[time_step:]

    assert len(units) == len(labels)
    assert len(units) == len(dates)

    options = dict(dtype=data_sets.train.dtype, seed=data_sets.train.seed, column_number=data_sets.column_number,
                   class_number=data_sets.class_number, batch_size=data_sets.batch_size)

    train = DataSet(units[:train_size], labels[:train_size], dates[:train_size], **options)
    test = DataSet(units[train_size:], labels[train_size:], dates[train_size:], **options)

    return DATASETS(train=train, test=test, column_number=data_sets.column_number,
                    class_number=data_sets.class_number, batch_size=data_sets.batch_size)


def dataframe_to_recurrent_ndarray(x, time_step):
    recurrent_panel = []
    for i in tqdm(range(0, len(x) - time_step)):
        recurrent_frame = []
        for index, values in x[i:i + time_step].iterrows():
            recurrent_frame.append(np.asarray(values))

        recurrent_panel.append(np.asarray(recurrent_frame))

    return np.asarray(recurrent_panel)


# noinspection PyUnresolvedReferences
def read_data(company_code,
              test_start_date=datetime(2017, 8, 1),
              shuffle=True,
              batch_size=100,
              dtype=dtypes.float32,
              seed=None):
    stock_masters = get_stock_master(company_code)
    stock_minute_prices = get_stock_minute_prices(stock_masters)
    stock_minute_prices.loc[:, 'next_close'] = stock_minute_prices['close'].shift(-1)
    stock_minute_prices.loc[:, 'label'] = (stock_minute_prices['close'] < stock_minute_prices['next_close']).astype(int)
    stock_minute_prices = stock_minute_prices.drop(columns=['next_close'])
    stock_minute_prices = stock_minute_prices.dropna()

    assert len(stock_minute_prices) > 0

    if shuffle:
        stock_minute_prices = stock_minute_prices.sample(frac=1)

    column_number = len(stock_minute_prices.columns) - 1
    units = stock_minute_prices.loc[:, stock_minute_prices.columns != 'label']
    units = pd.DataFrame(MinMaxScaler().fit_transform(units))
    stock_minute_prices = stock_minute_prices.reset_index()
    labels = stock_minute_prices.loc[:, ['label']]
    dates = stock_minute_prices.loc[:, ['date']]

    train_units = units[stock_minute_prices['date'] < test_start_date]
    train_labels = labels[stock_minute_prices['date'] < test_start_date]['label']
    train_dates = dates[stock_minute_prices['date'] < test_start_date]['date']
    test_units = units[stock_minute_prices['date'] >= test_start_date]
    test_labels = labels[stock_minute_prices['date'] >= test_start_date]['label']
    test_dates = dates[stock_minute_prices['date'] >= test_start_date]['date']

    options = dict(dtype=dtype, seed=seed, column_number=column_number, class_number=2, batch_size=batch_size)

    train = DataSet(train_units, train_labels, train_dates, **options)
    test = DataSet(test_units, test_labels, test_dates, **options)

    return DATASETS(train=train, test=test, column_number=column_number, class_number=2, batch_size=batch_size)
