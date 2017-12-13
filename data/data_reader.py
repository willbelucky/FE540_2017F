# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 3.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from pandas import DataFrame

idx = pd.IndexSlice

STOCK_MASTER_TABLE = 'stock_master'
STOCK_PRICE_TABLE = 'stock_price'
NAVER_FINANCE_FORUM_TABLE = 'naver_finance_forum'
NAVER_FINANCE_FORUM_STAT_TABLE = 'naver_finance_forum_stat'

CACHE = {
    STOCK_MASTER_TABLE: None,
    STOCK_PRICE_TABLE: None,
    NAVER_FINANCE_FORUM_TABLE: None,
    NAVER_FINANCE_FORUM_STAT_TABLE: None
}

# Set your working directory FE540_2017F.
# For example, /Users/willbe/PycharmProjects/FE540_2017F
DATA_DIR = os.getcwd().replace(chr(92), '/') + '/data/'


def get_cached_table(table_name: str, index=None, parse_dates=None) -> DataFrame or None:
    """

    :param table_name: (str)
    :param index: (None or list[str])
    :param parse_dates: (None or list[str])
    :return cached_table: (DataFrame)
    """
    hdf_file = table_name + '.h5'
    csv_file = table_name + '.csv'
    if CACHE[table_name] is None:
        # Read h5 raw data file.
        if Path(DATA_DIR + hdf_file).exists():
            CACHE[table_name] = pd.read_hdf(DATA_DIR + hdf_file, 'table')
            if parse_dates is not None:
                for parse_date in parse_dates:
                    CACHE[table_name][parse_date] = pd.to_datetime(CACHE[table_name][parse_date])
        # If there is no h5 file, read excel raw data file.
        else:
            CACHE[table_name] = pd.read_csv(DATA_DIR + csv_file,
                                            parse_dates=parse_dates, low_memory=False)
            CACHE[table_name].to_hdf(hdf_file, 'table')
            print('Create {}'.format(hdf_file))

        if index is not None:
            CACHE[table_name] = CACHE[table_name].set_index(index)

        if table_name == STOCK_PRICE_TABLE:
            CACHE[STOCK_PRICE_TABLE]['adj_close'] = CACHE[STOCK_PRICE_TABLE]['market_capitalization'] / \
                                                    CACHE[STOCK_PRICE_TABLE]['listed_stocks_number']
            CACHE[STOCK_PRICE_TABLE]['adj_open'] = CACHE[STOCK_PRICE_TABLE]['adj_close'] / CACHE[STOCK_PRICE_TABLE][
                'close'] * CACHE[STOCK_PRICE_TABLE]['open']

    cached_table = CACHE[table_name]

    return cached_table


def _get_stock_master_table() -> DataFrame or None:
    """

    :return stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    """
    stock_masters = get_cached_table(STOCK_MASTER_TABLE, index=['code'])
    return stock_masters


def _get_stock_price_table() -> DataFrame or None:
    """

    :return stock_prices: (DataFrame)
        index   code                    | (str) 6 digits number string representing a company.
                date                    | (datetime) The created date.
        column  volume                  | (int) The number of traded stocks of a day.
                open                    | (int) The first price of a day.
                high                    | (int) The highest price of a day.
                low                     | (int) The lowest price of a day.
                close                   | (int) The final price of a day.
                market_capitalization   | (int) The market capitalization of a company.
                listed_stocks_number    | (int) The number of listed stocks of a company.
                adj_close               | (float) The adjusted close price.
                adj_open                | (float) The adjusted open price.
    """
    stock_prices = get_cached_table(STOCK_PRICE_TABLE, index=['code', 'date'], parse_dates=['date'])
    return stock_prices


def _get_naver_finance_forum_table() -> DataFrame or None:
    """

    :return naver_finance_forums: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date and time.
                writer          | (str) The writer of the forum.
        column  title           | (str) The title.
                opinion         | (str) One of ['의견없음', '강력매수', '매수', '중립', '매도', '강력매도']
                hit             | (int) How many views of this post is.
                agreement       | (int) The number of agreements.
                disagreement    | (int) The number of disagreements.
    """
    naver_finance_forums = get_cached_table(NAVER_FINANCE_FORUM_TABLE, index=['code', 'date', 'writer'],
                                            parse_dates=['date'])
    return naver_finance_forums


def _get_naver_finance_forum_stat_table() -> DataFrame or None:
    """

    :return naver_finance_forum_stats: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date.
        column  count   | (int) The number of posts.
    """
    naver_finance_forum_stats = get_cached_table(NAVER_FINANCE_FORUM_STAT_TABLE, index=['code', 'date'],
                                                 parse_dates=['date'])
    return naver_finance_forum_stats


def get_stock_master(code: str) -> DataFrame:
    """

    :param code: (str) 6 digits number string representing a company.

    :return stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    """
    stock_masters = _get_stock_master_table()
    stock_masters = stock_masters[stock_masters.index.values == code]
    return stock_masters


def get_stock_masters(codes: List[str] = None) -> DataFrame:
    """

    :param codes: ([str]) 6 digits number strings representing companies.

    :return stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    """
    stock_masters = _get_stock_master_table()
    if codes is not None:
        stock_masters = stock_masters.loc[codes]
    return stock_masters


def get_stock_prices(stock_masters: DataFrame) -> DataFrame:
    """

    :param stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.

    :return stock_prices: (DataFrame)
        index   code                    | (str) 6 digits number string representing a company.
                date                    | (datetime) The created date.
        column  volume                  | (int) The number of traded stocks of a day.
                open                    | (int) The first price of a day.
                high                    | (int) The highest price of a day.
                low                     | (int) The lowest price of a day.
                close                   | (int) The final price of a day.
                market_capitalization   | (int) The market capitalization of a company.
                listed_stocks_number    | (int) The number of listed stocks of a company.
                adj_close               | (float) The adjusted close price.
                adj_open                | (float) The adjusted open price.
    """
    stock_prices = _get_stock_price_table()
    stock_prices = stock_prices.loc[idx[stock_masters.index.values, :], :]
    return stock_prices


def get_naver_finance_forums(stock_masters: DataFrame = None, from_date: datetime = datetime(2016, 10, 31),
                             to_date: datetime = datetime(2017, 10, 31)) -> DataFrame:
    """

    :param stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    :param from_date: (datetime) The start date.
    :param to_date: (datetime) The end date.

    :return naver_finance_forums: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date and time.
                writer          | (str) The writer of the forum.
        column  title           | (str) The title.
                opinion         | (str) One of ['의견없음', '강력매수', '매수', '중립', '매도', '강력매도']
                hit             | (int) How many views of this post is.
                agreement       | (int) The number of agreements.
                disagreement    | (int) The number of disagreements.
    """
    _to_date = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
    naver_finance_forums = _get_naver_finance_forum_table()
    if stock_masters is None:
        naver_finance_forums = naver_finance_forums.loc[idx[:, from_date:_to_date, :], :]
    else:
        naver_finance_forums = naver_finance_forums.loc[idx[stock_masters.index.values, from_date:_to_date, :], :]
    return naver_finance_forums


def get_naver_finance_forum_stats(stock_masters: DataFrame = None, from_date: datetime = datetime(2016, 10, 31),
                                  to_date: datetime = datetime(2017, 10, 31)) -> DataFrame:
    """

    :param stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    :param from_date: (datetime) The start date.
    :param to_date: (datetime) The end date.

    :return naver_finance_forum_stats: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date.
        column  count   | (int) The number of posts.
    """
    _to_date = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
    naver_finance_forum_stats = _get_naver_finance_forum_stat_table()
    if stock_masters is None:
        naver_finance_forum_stats = naver_finance_forum_stats.loc[idx[:, from_date:_to_date], :]
    else:
        naver_finance_forum_stats = naver_finance_forum_stats.loc[
                                    idx[stock_masters.index.values, from_date:_to_date], :]
    return naver_finance_forum_stats
