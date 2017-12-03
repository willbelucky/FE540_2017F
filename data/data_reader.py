# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 3.
"""
from datetime import datetime

import pandas as pd

idx = pd.IndexSlice

STOCK_MASTER_TABLE = 'stock_master.csv'
STOCK_PRICE_TABLE = 'stock_price.csv'
NAVER_FINANCE_FORUM_TABLE = 'naver_finance_forum.csv'
NAVER_FINANCE_FORUM_STAT_TABLE = 'naver_finance_forum_stat.csv'

CACHE = {
    STOCK_MASTER_TABLE: None,
    STOCK_PRICE_TABLE: None,
    NAVER_FINANCE_FORUM_TABLE: None,
    NAVER_FINANCE_FORUM_STAT_TABLE: None
}


def _get_stock_master_table():
    """

    :return stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    """
    if CACHE[STOCK_MASTER_TABLE] is None:
        CACHE[STOCK_MASTER_TABLE] = pd.read_csv(STOCK_MASTER_TABLE, index_col=['code'], low_memory=False)
    stock_masters = CACHE[STOCK_MASTER_TABLE]
    return stock_masters


def _get_stock_price_table():
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
    if CACHE[STOCK_PRICE_TABLE] is None:
        CACHE[STOCK_PRICE_TABLE] = pd.read_csv(STOCK_PRICE_TABLE, index_col=['code', 'date'], parse_dates=['date'],
                                               low_memory=False)
        CACHE[STOCK_PRICE_TABLE]['adj_close'] = \
            CACHE[STOCK_PRICE_TABLE]['market_capitalization'] / CACHE[STOCK_PRICE_TABLE]['listed_stocks_number']
        CACHE[STOCK_PRICE_TABLE]['adj_open'] = \
            CACHE[STOCK_PRICE_TABLE]['adj_close'] / CACHE[STOCK_PRICE_TABLE]['close'] * CACHE[STOCK_PRICE_TABLE]['open']
    stock_prices = CACHE[STOCK_PRICE_TABLE]
    return stock_prices


def _get_naver_finance_forum_table():
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
    if CACHE[NAVER_FINANCE_FORUM_TABLE] is None:
        CACHE[NAVER_FINANCE_FORUM_TABLE] = pd.read_csv(NAVER_FINANCE_FORUM_TABLE, index_col=['code', 'date', 'writer'],
                                                       parse_dates=['date'], low_memory=False)
    naver_finance_forums = CACHE[NAVER_FINANCE_FORUM_TABLE]
    return naver_finance_forums


def _get_naver_finance_forum_stat_table():
    """

    :return naver_finance_forum_stats: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date.
        column  count   | (int) The number of posts.
    """
    if CACHE[NAVER_FINANCE_FORUM_STAT_TABLE] is None:
        CACHE[NAVER_FINANCE_FORUM_STAT_TABLE] = pd.read_csv(NAVER_FINANCE_FORUM_STAT_TABLE, index_col=['code', 'date'],
                                                            parse_dates=['date'], low_memory=False).sort_index()
    naver_finance_forum_stats = CACHE[NAVER_FINANCE_FORUM_STAT_TABLE]
    return naver_finance_forum_stats


def get_all_stock_masters():
    """

    :return stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    """
    stock_masters = _get_stock_master_table()
    return stock_masters


def get_stock_master(code):
    """

    :param code: (str) 6 digits number string representing a company.

    :return stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    """
    stock_masters = _get_stock_master_table()
    stock_masters = stock_masters[stock_masters.index.values == code]
    return stock_masters


def get_stock_masters(codes):
    """

    :param codes: ([str]) 6 digits number strings representing companies.

    :return stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    """
    stock_masters = _get_stock_master_table()
    stock_masters = stock_masters.loc[codes]
    return stock_masters


def get_stock_prices(stock_masters):
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


def get_naver_finance_forums(stock_masters, from_date, to_date):
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
    naver_finance_forums = naver_finance_forums.loc[idx[stock_masters.index.values, from_date:_to_date, :], :]
    return naver_finance_forums


def get_naver_finance_forum_stats(stock_masters, from_date, to_date):
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
    naver_finance_forum_stats = naver_finance_forum_stats.loc[idx[stock_masters.index.values, from_date:_to_date], :]
    return naver_finance_forum_stats
