# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 22.
"""
import argparse
import os
import sys
from datetime import datetime

import tensorflow as tf

from quantitative_behavior.classifying_lstm import run_training
from quantitative_behavior.data_feeder import read_data

FLAGS = None


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training(flags=FLAGS, data_sets=read_data(test_start_date=datetime(2017, 8, 1), shuffle=False))


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
        default=None,
        help='The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--time_step',
        type=int,
        default=10,
        help='Number of time window.'
    )
    parser.add_argument(
        '--hidden_units',
        nargs='+',
        type=int,
        default=[128],
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'images/'),
        help='Directory to put the image.'
    )
    parser.add_argument(
        '--excel_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'excels/'),
        help='Directory to put the excel file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)