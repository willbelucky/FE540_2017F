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

from high_frequency.classifying_lstm import run_training
from high_frequency.data_feeder import read_data
from data_dealer.data_reader import get_stock_masters

FLAGS = None


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

    if FLAGS.company_codes is None:
        stock_masters = get_stock_masters()
        for company_code in stock_masters.index.values:
            try:
                run_training(flags=FLAGS,
                             data_sets=read_data(company_code=company_code,
                                                 test_start_date=datetime(2017, 8, 1),
                                                 shuffle=False,
                                                 batch_size=FLAGS.batch_size))
            except AssertionError as ae:
                print(ae)
    else:
        for company_code in FLAGS.company_codes:
            run_training(flags=FLAGS,
                         data_sets=read_data(company_code=company_code,
                                             test_start_date=datetime(2017, 8, 1),
                                             shuffle=False,
                                             batch_size=FLAGS.batch_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--company_codes',
        nargs='+',
        type=str,
        default=None,
        help='Codes of companies.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=100,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--time_step',
        type=int,
        default=1000,
        help='Number of time window.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Number of batch_size.'
    )
    parser.add_argument(
        '--hidden_units',
        nargs='+',
        type=int,
        default=[512, 512],
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'high_frequency/images/'),
        help='Directory to put the image.'
    )
    parser.add_argument(
        '--excel_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'high_frequency/excels/'),
        help='Directory to put the excel file.'
    )
    parser.add_argument(
        '--html_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'high_frequency/htmls/'),
        help='Directory to put the html file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getcwd(), 'high_frequency/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)