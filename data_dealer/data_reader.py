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
STOCK_MINUTE_PRICE_TABLE = 'stock_minute_price'
STOCK_VOLUME_TABLE = 'stock_volume'
NAVER_FINANCE_FORUM_TABLE = 'naver_finance_forum'
NAVER_FINANCE_FORUM_STAT_TABLE = 'naver_finance_forum_stat'
PROFIT_TITLE_TABLE = 'profit_title'
VOLATILITY_TITLE_TABLE = 'volatility_title'
WORD_PACK_TABLE = 'word_pack'
QUANTITATIVE_BEHAVIOR_TABLE = 'quantitative_behavior'
HIGH_FREQUENCY_VOLATILITY = 'high_frequency_volatility'
HIGH_FREQUENCY_PROFIT = 'high_frequency_profit'

CACHE = {
    STOCK_MASTER_TABLE: None,
    STOCK_PRICE_TABLE: None,
    STOCK_MINUTE_PRICE_TABLE: None,
    STOCK_VOLUME_TABLE: None,
    NAVER_FINANCE_FORUM_TABLE: None,
    NAVER_FINANCE_FORUM_STAT_TABLE: None,
    PROFIT_TITLE_TABLE: None,
    VOLATILITY_TITLE_TABLE: None,
    WORD_PACK_TABLE: None,
    QUANTITATIVE_BEHAVIOR_TABLE: None,
    HIGH_FREQUENCY_VOLATILITY: None,
    HIGH_FREQUENCY_PROFIT: None,
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
            # Drop duplicates except for the last occurrence.
            if index is not None:
                CACHE[table_name] = CACHE[table_name].drop_duplicates(subset=index, keep='last')
            # Save as a h5 file.
            CACHE[table_name].to_hdf(DATA_DIR + hdf_file, 'table')
            print('Create {}'.format(DATA_DIR + hdf_file))

        if index is not None:
            CACHE[table_name] = CACHE[table_name].set_index(index)
            CACHE[table_name] = CACHE[table_name].sort_index()

        # Treat stock_price specially.
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


def _get_stock_minute_price_table() -> DataFrame or None:
    """

    :return stock_minute_prices: (DataFrame)
        index   code                    | (str) 6 digits number string representing a company.
                date                    | (datetime) The created date.
        column  volume                  | (int) The number of traded stocks of a day.
                open                    | (int) The first price of a day.
                high                    | (int) The highest price of a day.
                low                     | (int) The lowest price of a day.
                close                   | (int) The final price of a day.
    """
    stock_minute_prices = get_cached_table(STOCK_MINUTE_PRICE_TABLE, index=['code', 'date'], parse_dates=['date'])
    return stock_minute_prices


def _get_stock_volume_table() -> DataFrame or None:
    """

    :return stock_volumes: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date.
        column  personal        | (int)
                national        | (int)
                investment      | (int)
                total_org       | (int)
                other_finance   | (int)
                other_law       | (int)
                other_foreign   | (int)
                foreign         | (int)
                insurance       | (int)
                pef             | (int)
                pension         | (int)
                total_foreign   | (int)
                bank            | (int)
                trust           | (int)
    """
    stock_volumes = get_cached_table(STOCK_VOLUME_TABLE, index=['code', 'date'], parse_dates=['date'])
    stock_volumes = stock_volumes.reset_index()
    stock_volumes['code'] = stock_volumes['code'].apply(str)
    stock_volumes['code'] = stock_volumes['code'].str.zfill(6)
    stock_volumes = stock_volumes.set_index(['code', 'date'])
    stock_volumes = stock_volumes.sort_index()
    return stock_volumes


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


def _get_word_pack_table() -> DataFrame or None:
    """

    :return word_pack: (DataFrame)
        column  code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
                word    | (str) A word.
    """
    word_pack = get_cached_table(WORD_PACK_TABLE, parse_dates=['date'])
    return word_pack


def _get_profit_title_table() -> DataFrame or None:
    """

    :return profit_titles: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
        column  title   | (str) The title.
                label   | (int) If stock profit is positive, label is 1, else label is 0.
    """
    profit_titles = get_cached_table(PROFIT_TITLE_TABLE, index=['code', 'date', 'writer'], parse_dates=['date'])
    return profit_titles


def _get_volatility_title_table() -> DataFrame or None:
    """

    :return volatility_titles: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
        column  title   | (str) The title.
                label   | (int) If stock profit is positive, label is 1, else label is 0.
    """
    volatility_titles = get_cached_table(VOLATILITY_TITLE_TABLE, index=['code', 'date', 'writer'], parse_dates=['date'])
    return volatility_titles


def _get_quantitative_behavior_table() -> DataFrame or None:
    """

    :return quantitative_behaviors: (DataFrame)
        index   code                | (str) 6 digits number string representing a company.
                date                | (datetime) The created date and time.
        column  adj_close_1/5       | (float)
                personal_1/5        | (float)
                national_1/5        | (float)
                investment_1/5      | (float)
                total_org_1/5       | (float)
                other_finance_1/5   | (float)
                other_law_1/5       | (float)
                other_foreign_1/5   | (float)
                foreign_1/5         | (float)
                insurance_1/5       | (float)
                pef_1/5             | (float)
                pension_1/5         | (float)
                total_foreign_1/5   | (float)
                bank_1/5            | (float)
                trust_1/5           | (float)
                count_1/5           | (float) The number of posts.
                adj_close_5/20      | (float)
                personal_5/20       | (float)
                national_5/20       | (float)
                investment_5/20     | (float)
                total_org_5/20      | (float)
                other_finance_5/20  | (float)
                other_law_5/20      | (float)
                other_foreign_5/20  | (float)
                foreign_5/20        | (float)
                insurance_5/20      | (float)
                pef_5/20            | (float)
                pension_5/20        | (float)
                total_foreign_5/20  | (float)
                bank_5/20           | (float)
                trust_5/20          | (float)
                count_5/20          | (float) The number of posts.
                adj_close_5/20      | (float)
                personal_20/60      | (float)
                national_20/60      | (float)
                investment_20/60    | (float)
                total_org_20/60     | (float)
                other_finance_20/60 | (float)
                other_law_20/60     | (float)
                other_foreign_20/60 | (float)
                foreign_20/60       | (float)
                insurance_20/60     | (float)
                pef_20/60           | (float)
                pension_20/60       | (float)
                total_foreign_20/60 | (float)
                bank_20/60          | (float)
                trust_20/60         | (float)
                count_20/60         | (float) The number of posts.
                label               | (int) If stock profit is positive, label is 1, else label is 0.
    """
    quantitative_behaviors = get_cached_table(QUANTITATIVE_BEHAVIOR_TABLE, index=['code', 'date'], parse_dates=['date'])
    return quantitative_behaviors


def _get_high_frequency_volatility_table() -> DataFrame or None:
    """

    :return high_frequency_volatilities: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date.
        column  volume          | (int) The number of traded stocks of a day.
                open            | (int) The first price of a day.
                high            | (int) The highest price of a day.
                low             | (int) The lowest price of a day.
                close           | (int) The final price of a day.
                volatility      | (float) The volatility of 5 minutes.
                label           | (float) The next 5 minutes volatility.
    """
    high_frequency_volatilities = get_cached_table(HIGH_FREQUENCY_VOLATILITY, index=['code', 'date'],
                                                   parse_dates=['date'])
    return high_frequency_volatilities


def _get_high_frequency_profit_table() -> DataFrame or None:
    """

    :return high_frequency_profits: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date.
        column  volume          | (int) The number of traded stocks of a day.
                open            | (int) The first price of a day.
                high            | (int) The highest price of a day.
                low             | (int) The lowest price of a day.
                close           | (int) The final price of a day.
                label           | (int) If the current close price is lower than the next close price,
                                        a label is 1. Else, a label is 0.
    """
    high_frequency_profits = get_cached_table(HIGH_FREQUENCY_PROFIT, index=['code', 'date'],
                                              parse_dates=['date'])
    return high_frequency_profits


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
    # assert len(stock_prices) > 0, 'There is no stock price of {}'.format(stock_masters.index.values)
    return stock_prices


def get_stock_minute_prices(stock_masters: DataFrame, from_date: datetime = datetime(2016, 11, 1),
                            to_date: datetime = datetime(2017, 10, 31)) -> DataFrame:
    """

    :param stock_masters: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
        column  name    | (str) The name of the company.
    :param from_date: (datetime) The start date.
    :param to_date: (datetime) The end date.

    :return stock_minute_prices: (DataFrame)
        index   code                    | (str) 6 digits number string representing a company.
                date                    | (datetime) The created date.
        column  volume                  | (int) The number of traded stocks of a day.
                open                    | (int) The first price of a day.
                high                    | (int) The highest price of a day.
                low                     | (int) The lowest price of a day.
                close                   | (int) The final price of a day.
    """
    _to_date = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
    stock_minute_prices = _get_stock_minute_price_table()
    stock_minute_prices = stock_minute_prices.loc[idx[stock_masters.index.values, from_date:_to_date], :]
    # assert len(stock_minute_prices) > 0, 'There is no stock minute price of {} from {}, to {}'.format(
    #     stock_masters.index.values, from_date, _to_date)
    return stock_minute_prices


def get_stock_volumes(stock_masters: DataFrame, from_date: datetime = datetime(2016, 11, 1),
                      to_date: datetime = datetime(2017, 10, 31)) -> DataFrame:
    """

    :return stock_volumes: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date.
        column  personal        | (int)
                national        | (int)
                investment      | (int)
                total_org       | (int)
                other_finance   | (int)
                other_law       | (int)
                other_foreign   | (int)
                foreign         | (int)
                insurance       | (int)
                pef             | (int)
                pension         | (int)
                total_foreign   | (int)
                bank            | (int)
                trust           | (int)
    """
    _to_date = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
    stock_volumes = _get_stock_volume_table()
    stock_volumes = stock_volumes.loc[idx[stock_masters.index.values, from_date:_to_date], :]
    return stock_volumes


def get_naver_finance_forums(stock_masters: DataFrame = None, from_date: datetime = datetime(2016, 11, 1),
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


def get_naver_finance_forum_stats(stock_masters: DataFrame = None, from_date: datetime = datetime(2016, 11, 1),
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


def get_word_pack():
    """

    :return word_pack: (DataFrame)
        column  code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
                word    | (str) A word.
    """
    word_pack = _get_word_pack_table()
    return word_pack


def get_profit_titles(stock_masters: DataFrame = None, from_date: datetime = datetime(2016, 11, 1),
                      to_date: datetime = datetime(2017, 10, 31)) -> DataFrame:
    """

    :return profit_titles: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
        column  title   | (str) The title.
                label   | (int) If stock profit is positive, label is 1, else label is 0.
    """
    _to_date = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
    profit_titles = _get_profit_title_table()
    if stock_masters is None:
        profit_titles = profit_titles.loc[idx[:, from_date:_to_date, :], :]
    else:
        profit_titles = profit_titles.loc[idx[stock_masters.index.values, from_date:_to_date, :], :]
    return profit_titles


def get_volatility_titles(stock_masters: DataFrame = None, from_date: datetime = datetime(2016, 11, 1),
                          to_date: datetime = datetime(2017, 10, 31)) -> DataFrame:
    """

    :return volatility_titles: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
        column  title   | (str) The title.
                label   | (int) If stock profit is positive, label is 1, else label is 0.
    """
    _to_date = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)
    volatility_titles = _get_volatility_title_table()
    if stock_masters is None:
        volatility_titles = volatility_titles.loc[idx[:, from_date:_to_date, :], :]
    else:
        volatility_titles = volatility_titles.loc[idx[stock_masters.index.values, from_date:_to_date, :], :]
    return volatility_titles


def get_quantitative_behaviors() -> DataFrame:
    """

    :return quantitative_behaviors: (DataFrame)
        index   code                | (str) 6 digits number string representing a company.
                date                | (datetime) The created date and time.
        column  adj_close_1/5       | (float)
                personal_1/5        | (float)
                national_1/5        | (float)
                investment_1/5      | (float)
                total_org_1/5       | (float)
                other_finance_1/5   | (float)
                other_law_1/5       | (float)
                other_foreign_1/5   | (float)
                foreign_1/5         | (float)
                insurance_1/5       | (float)
                pef_1/5             | (float)
                pension_1/5         | (float)
                total_foreign_1/5   | (float)
                bank_1/5            | (float)
                trust_1/5           | (float)
                count_1/5           | (float) The number of posts.
                adj_close_5/20      | (float)
                personal_5/20       | (float)
                national_5/20       | (float)
                investment_5/20     | (float)
                total_org_5/20      | (float)
                other_finance_5/20  | (float)
                other_law_5/20      | (float)
                other_foreign_5/20  | (float)
                foreign_5/20        | (float)
                insurance_5/20      | (float)
                pef_5/20            | (float)
                pension_5/20        | (float)
                total_foreign_5/20  | (float)
                bank_5/20           | (float)
                trust_5/20          | (float)
                count_5/20          | (float) The number of posts.
                adj_close_5/20      | (float)
                personal_20/60      | (float)
                national_20/60      | (float)
                investment_20/60    | (float)
                total_org_20/60     | (float)
                other_finance_20/60 | (float)
                other_law_20/60     | (float)
                other_foreign_20/60 | (float)
                foreign_20/60       | (float)
                insurance_20/60     | (float)
                pef_20/60           | (float)
                pension_20/60       | (float)
                total_foreign_20/60 | (float)
                bank_20/60          | (float)
                trust_20/60         | (float)
                count_20/60         | (float) The number of posts.
                label               | (int) If stock profit is positive, label is 1, else label is 0.
    """
    quantitative_behaviors = _get_quantitative_behavior_table()
    return quantitative_behaviors


def get_high_frequency_volatilities(stock_masters: DataFrame = None) -> DataFrame:
    """

    :return high_frequency_volatilities: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date.
        column  volume          | (int) The number of traded stocks of a day.
                open            | (int) The first price of a day.
                high            | (int) The highest price of a day.
                low             | (int) The lowest price of a day.
                close           | (int) The final price of a day.
                volatility      | (float) The volatility of 5 minutes.
                label           | (float) The next 5 minutes volatility.
    """
    high_frequency_volatilities = _get_high_frequency_volatility_table()
    if stock_masters is None:
        high_frequency_volatilities = high_frequency_volatilities.loc[idx[:, :], :]
    else:
        high_frequency_volatilities = high_frequency_volatilities.loc[idx[stock_masters.index.values, :], :]
    return high_frequency_volatilities


def get_high_frequency_profits(stock_masters: DataFrame = None) -> DataFrame:
    """

    :return high_frequency_profits: (DataFrame)
        index   code            | (str) 6 digits number string representing a company.
                date            | (datetime) The created date.
        column  volume          | (int) The number of traded stocks of a day.
                open            | (int) The first price of a day.
                high            | (int) The highest price of a day.
                low             | (int) The lowest price of a day.
                close           | (int) The final price of a day.
                volatility      | (float) The volatility of 5 minutes.
                label           | (int) If the current close price is lower than the next close price,
                                        a label is 1. Else, a label is 0.
    """
    high_frequency_profits = _get_high_frequency_profit_table()
    if stock_masters is None:
        high_frequency_profits = high_frequency_profits.loc[idx[:, :], :]
    else:
        high_frequency_profits = high_frequency_profits.loc[idx[stock_masters.index.values, :], :]
    return high_frequency_profits
