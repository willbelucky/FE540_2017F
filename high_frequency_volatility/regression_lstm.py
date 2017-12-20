# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 27.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from high_frequency_volatility.data_feeder import to_recurrent_data
from stats.statistical_predictor import ewma, arima
from util.telegram import send_message

warnings.filterwarnings("ignore")


def lstm_cell(hidden_unit, dropout):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_unit, state_is_tuple=True, activation=tf.tanh)
    if dropout is not None:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    return cell


def inference(units, hidden_units, column_number, class_number, batch_size, dropout=None):
    """Build the mnist_example model up to where it may be used for inference.

    Args:
      units: Units placeholder, from inputs().
      hidden_units: Size of the hidden layers.
      column_number: Size of the input columns.
      class_number: Size of the output classes.
      batch_size: Size of a batch
      dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(hidden_unit, dropout) for hidden_unit in hidden_units])
    outputs, _states = tf.nn.dynamic_rnn(stacked_lstm, units, dtype=tf.float32)
    dense = tf.layers.dense(outputs[:, -1], batch_size)
    predictions = tf.layers.dense(dense, column_number)

    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([column_number, class_number],
                                                  stddev=1.0 / math.sqrt(float(column_number))), name='weights')
        biases = tf.Variable(tf.zeros([class_number]), name='biases')
        logits = tf.matmul(predictions, weights) + biases
    return logits


def do_loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float32 - [batch_size, NUM_CLASSES].
      labels: Labels tensor, float32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    return tf.reduce_sum(tf.square(logits - labels))


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def placeholder_inputs(batch_size, time_step, column_number):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
      time_step: The number of time steps.
      column_number: Size of the input columns.
    Returns:
      units_placeholder: Units placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # unit and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    units_placeholder = tf.placeholder(tf.float32, [batch_size, time_step, column_number])
    labels_placeholder = tf.placeholder(tf.float32, [batch_size])
    return units_placeholder, labels_placeholder


def fill_feed_dict(data_set, units_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of units and labels, from data.read_data_sets()
      units_pl: The units placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    units_feed, labels_feed, dates_feed = data_set.next_batch(shuffle=False)
    feed_dict = {
        units_pl: units_feed,
        labels_pl: labels_feed,
    }
    return feed_dict, labels_feed, dates_feed


def do_eval(sess,
            logits,
            units_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      logits:
      units_placeholder: The units placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of units and labels to evaluate, from
        data.read_data_sets().
    """
    # And run one epoch of eval.
    feed_dict, labels_feed, dates_feed = fill_feed_dict(data_set,
                                                        units_placeholder,
                                                        labels_placeholder)
    lstm_predictions = sess.run(logits, feed_dict=feed_dict)
    lstm_mean_squared_error = mean_squared_error(labels_feed, lstm_predictions)
    return lstm_mean_squared_error


def to_excel(dataframe, dir, file_name):
    writer = pd.ExcelWriter(dir + file_name + '.xlsx')
    dataframe.to_excel(writer)


def run_training(company_name, flags, data_sets):
    """Train mnist_example for a number of steps."""

    file_name = 'high_frequency_volatility_{}_{}_{}_{}_{}_{}'.format(company_name,
                                                                     flags.learning_rate,
                                                                     flags.dropout,
                                                                     flags.max_steps,
                                                                     flags.time_step,
                                                                     flags.hidden_units)

    # If the result file exists, do not run.
    if Path(flags.image_dir + file_name + '.png').exists():
        return

    data_sets = to_recurrent_data(data_sets, flags.time_step, company_name)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the units and labels.
        units_placeholder, labels_placeholder = placeholder_inputs(data_sets.batch_size, flags.time_step,
                                                                   data_sets.column_number)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(units_placeholder,
                           flags.hidden_units,
                           data_sets.column_number,
                           data_sets.class_number,
                           data_sets.batch_size,
                           flags.dropout)

        # Add to the Graph the Ops for loss calculation.
        loss = do_loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, flags.learning_rate)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        sess.run(init)

        # Initialize a dictionary for save test result temporally.
        results = {
            'step': [],
            'train_MSE': [],
            'test_MSE': [],
        }

        print("\t".join(['learning_rate', 'max_steps', 'hidden_units']))
        print("{:f}\t{:d}\t{}".format(flags.learning_rate, flags.max_steps, flags.hidden_units))
        print("")
        print(" ".join(['step', 'loss_value', 'training_MSE', 'test_MSE']))
        # Start the training loop.
        for step in range(flags.max_steps + 1):

            # Fill a feed dictionary with the actual set of units and labels
            # for this particular training step.
            feed_dict, _, _ = fill_feed_dict(data_sets.train,
                                             units_placeholder,
                                             labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            # Save a checkpoint and evaluate the model periodically.
            if step % (flags.max_steps / 10) == 0:
                checkpoint_file = os.path.join(flags.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                training_mean_squared_error = do_eval(sess,
                                                      logits,
                                                      units_placeholder,
                                                      labels_placeholder,
                                                      data_sets.train)
                # Evaluate against the test set.
                test_mean_squared_error = do_eval(sess,
                                                  logits,
                                                  units_placeholder,
                                                  labels_placeholder,
                                                  data_sets.test)

                assert not np.isnan(training_mean_squared_error)
                assert not np.isnan(test_mean_squared_error)

                print("{:d}\t{:f}\t{:f}\t{:f}".format(step, loss_value, training_mean_squared_error,
                                                      test_mean_squared_error))
                results['step'].append(step)
                results['train_MSE'].append(training_mean_squared_error)
                results['test_MSE'].append(test_mean_squared_error)

        # Save the test result as an excel file.
        test_result = pd.DataFrame(results)
        test_result = test_result.set_index(keys=['step'])
        to_excel(test_result, flags.excel_dir, file_name)

        # Compare labels(targets) with predictions.
        feed_dict, labels_feed, dates_feed = fill_feed_dict(data_sets.test,
                                                            units_placeholder,
                                                            labels_placeholder)
        targets = labels_feed

        # ARIMA plot
        auto_arima = {
            '삼성전자': (2, 1, 0),
            '현대차': (5, 1, 0),
            'LG전자': (2, 1, 5),
            '한국항공우주': (5, 1, 5),
            '한국전력': (3, 1, 4),
            '현대백화점': (2, 1, 2),
            '카카오': (0, 1, 1),
        }
        if company_name in auto_arima.keys():
            p, d, q = auto_arima[company_name]
        else:
            p, d, q = 5, 1, 0
        arima_file_name = 'arima_{}_{}_{}_{}.h5'.format(company_name, p, d, q)
        if Path(flags.arima_dir + arima_file_name).exists():
            arima_prediction_df = pd.read_hdf(flags.arima_dir + arima_file_name, 'table')
            arima_predictions = arima_prediction_df['prediction'].tolist()
        else:
            history = data_sets.train.labels.tolist()
            arima_predictions = arima(history, targets, p, d, q)
            arima_prediction_df = pd.DataFrame(arima_predictions, columns=['prediction'])
            arima_prediction_df.to_hdf(flags.arima_dir + arima_file_name, 'table')

        # EWMA plot
        lambda_window = 30
        lambda_percent = 0.94
        ewma_file_name = 'ewma_{}_{}_{}.h5'.format(company_name, lambda_window, lambda_percent)
        if Path(flags.ewma_dir + ewma_file_name).exists():
            ewma_prediction_df = pd.read_hdf(flags.ewma_dir + ewma_file_name, 'table')
            ewma_predictions = ewma_prediction_df['prediction'].tolist()
        else:
            test_prices = pd.Series(data_sets.test.units[:, 0, 3])
            ewma_predictions = ewma(test_prices).tolist()
            ewma_prediction_df = pd.DataFrame(ewma_predictions, columns=['prediction'])
            ewma_prediction_df.to_hdf(flags.ewma_dir + ewma_file_name, 'table')

        # LSTM plot
        lstm_predictions = sess.run(logits, feed_dict)

        # CNN plot

        # Martingale plot
        martingale_predictions = np.concatenate((data_sets.train.labels[-1:], data_sets.test.labels[:-1]), axis=0)

        # calculate a mean error.
        arima_mean_error = mean_squared_error(targets, arima_predictions)
        lstm_mean_error = mean_squared_error(targets, lstm_predictions)
        martingale_mean_error = mean_squared_error(targets, martingale_predictions)

        # draw a test graph
        matplotlib.rc('font', family='NanumBarunGothicOTF')
        fig, ax = plt.subplots()
        ax.plot(dates_feed, targets, 'k', label='target', linewidth=1)
        ax.plot(dates_feed, martingale_predictions, 'y',
                label='Martingale, {:.4f}'.format(martingale_mean_error), linewidth=1)
        ax.plot(dates_feed, arima_predictions, 'g', label='ARIMA, {:.4f}'.format(arima_mean_error), linewidth=1)
        ax.plot(dates_feed, ewma_predictions, 'p', label='EWMA, {:.4f}'.format(arima_mean_error), linewidth=1)
        ax.plot(dates_feed, lstm_predictions, 'b', label='LSTM, {:.4f}'.format(lstm_mean_error), linewidth=1)
        ax.legend()

        plt.title(company_name)
        plt.xlabel('date')
        plt.ylabel('MSE')
        plt.savefig(flags.image_dir + file_name + '.png')
        plt.close()

        # Send a notice message to telegram.
        send_message(file_name + ' is done!')
