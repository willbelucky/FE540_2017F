# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 13.
"""
import argparse
import os
import sys

import tensorflow as tf

from word2vec.word_to_vector import run_training

from data_dealer.data_reader import get_titles


def data_reader():
    titles = get_titles()
    sentences = [words.split() for words in titles['title'].values]
    targets = titles['label'].tolist()

    return sentences, targets


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.excel_dir):
        tf.gfile.MakeDirs(FLAGS.excel_dir)
    if not tf.gfile.Exists(FLAGS.image_dir):
        tf.gfile.MakeDirs(FLAGS.image_dir)
    if not tf.gfile.Exists(FLAGS.html_dir):
        tf.gfile.MakeDirs(FLAGS.html_dir)
    sentences, targets = data_reader()
    run_training(flags=FLAGS, sentences=sentences, targets=targets, demonstration=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.4,
        help='The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='Number of embedding_size.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10000,
        help='Number of batch_size.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden_unit',
        type=int,
        default=256,
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--test_rate',
        type=float,
        default=0.0,
        help='Test rate. The portion of test set.'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'word2vec/images/'),
        help='Directory to put the image.'
    )
    parser.add_argument(
        '--excel_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'word2vec/excels/'),
        help='Directory to put the excel file.'
    )
    parser.add_argument(
        '--html_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'word2vec/htmls/'),
        help='Directory to put the html file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'word2vec/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
