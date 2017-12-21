# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 22.
"""
import argparse
import os
import sys

import tensorflow as tf

from data_dealer.data_reader import get_stock_masters
from daily_volatility.data_feeder import read_data
from daily_volatility.regression_lstm import run_training

FLAGS = None

FOLDER_DIR = 'daily_volatility/'


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
    if not tf.gfile.Exists(FLAGS.arima_dir):
        tf.gfile.MakeDirs(FLAGS.arima_dir)

    if FLAGS.company_names is None:
        stock_masters = get_stock_masters()
        for company_code, values in stock_masters.iterrows():
            print(company_code, values[0])
            try:
                run_training(values[0],
                             flags=FLAGS,
                             data_sets=read_data(company_name=values[0],
                                                 pca_alpha=FLAGS.pca_alpha,
                                                 shuffle=False))
            except AssertionError as ae:
                print(ae)
    else:
        for company_name in FLAGS.company_names:
            print(company_name)
            run_training(company_name,
                         flags=FLAGS,
                         data_sets=read_data(company_name=company_name,
                                             pca_alpha=FLAGS.pca_alpha,
                                             shuffle=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--company_names',
        nargs='+',
        type=str,
        default=[
            '한국전력',
            '현대백화점',
            '현대차',
            'LG전자',
            '삼성전자',
            '한국항공우주',
        ],
        help='Codes of companies.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.'
    )
    parser.add_argument(
        '--pca_alpha',
        type=float,
        default=0.95,
        help='PCA alpha. If pca_alpha is None, do not apply PCA.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--time_step',
        type=int,
        default=300,
        help='Number of time window.'
    )
    parser.add_argument(
        '--hidden_units',
        nargs='+',
        type=int,
        default=[256, 256],
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=os.path.join(os.getcwd(), FOLDER_DIR + 'images/'),
        help='Directory to put the image.'
    )
    parser.add_argument(
        '--excel_dir',
        type=str,
        default=os.path.join(os.getcwd(), FOLDER_DIR + 'excels/'),
        help='Directory to put the excel file.'
    )
    parser.add_argument(
        '--html_dir',
        type=str,
        default=os.path.join(os.getcwd(), FOLDER_DIR + 'htmls/'),
        help='Directory to put the html file.'
    )
    parser.add_argument(
        '--arima_dir',
        type=str,
        default=os.path.join(os.getcwd(), FOLDER_DIR + 'arimas/'),
        help='Directory to put the arima file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getcwd(), FOLDER_DIR + 'logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
