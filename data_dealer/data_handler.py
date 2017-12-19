# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 13.
"""
import os

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from data_dealer.data_reader import get_naver_finance_forums, get_stock_masters, get_stock_minute_prices, \
    get_stock_prices, get_naver_finance_forum_stats, get_stock_volumes, get_stock_master

idx = pd.IndexSlice

# Set your working directory FE540_2017F.
# For example, /Users/willbe/PycharmProjects/FE540_2017F
DATA_DIR = os.getcwd().replace(chr(92), '/') + '/data/'


def calculate_word_pack():
    """
    Split titles of naver_finance_forums to lists of words and flatten them.

    :return word_pack: (DataFrame)
        column  code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
                word    | (str) A word.
    """
    naver_finance_forums = get_naver_finance_forums()
    naver_finance_forums = naver_finance_forums['title']
    naver_finance_forums = naver_finance_forums.str.split()
    word_pack = []
    for (code, date, writer), words in tqdm(naver_finance_forums.iteritems(), desc='word_pack'):
        for word in words:
            word_pack.append({
                'code': code,
                'date': date,
                'writer': writer,
                'word': word,
            })

    word_pack = pd.DataFrame(word_pack)

    return word_pack


def calculate_profit_titles():
    """

    :return profit_titles: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
        column  title   | (str) The title.
                label   | (int) If stock profit is positive, label is 1, else label is 0.
    """
    stock_masters = get_stock_masters()
    merged_titles = pd.DataFrame()

    for i in tqdm(range(len(stock_masters)), desc='profit_title'):
        stock_master = stock_masters.iloc[i:i + 1]
        naver_finance_forums = get_naver_finance_forums(stock_master).reset_index()
        stock_minute_prices = get_stock_minute_prices(stock_master).reset_index()

        # If there is no stock minute prices of this company, pass.
        if stock_minute_prices.empty:
            continue

        minutes = pd.DataFrame(
            data=pd.date_range(start='2016/11/01 09:00:00', end='2017/11/01 23:59:00', freq='T'),
            columns=['date'])
        # Merge stock_minute_prices with period_minutes
        stock_minute_prices = pd.merge(stock_minute_prices, minutes, on=['date'], how='outer')
        stock_minute_prices = stock_minute_prices.set_index(['date'])
        stock_minute_prices = stock_minute_prices.sort_index()
        stock_minute_prices = stock_minute_prices.fillna(method='ffill')
        stock_minute_prices = stock_minute_prices.dropna()
        stock_minute_prices = stock_minute_prices.reset_index()

        # Merge naver_finance_forums with stock_minute_prices
        new_titles = pd.merge(naver_finance_forums, stock_minute_prices, on=['code', 'date'], how='inner')
        new_titles = new_titles.set_index(['code', 'date', 'writer'])
        new_titles['next_close'] = np.nan

        stock_prices = get_stock_prices(stock_master)
        for (code, date), row in stock_prices.iterrows():
            market_close_datetime = datetime(date.year, date.month, date.day, 15, 25, 00)
            try:
                selected_titles = new_titles.loc[idx[code, :market_close_datetime, :], :]
                selected_titles = selected_titles[selected_titles['next_close'].isnull()]
                selected_titles['next_close'] = row['close']
                try:
                    new_titles.update(selected_titles)
                except Exception:
                    raise Exception('code {} is duplicated!'.format(code))
            except KeyError:
                pass

        # Save new_titles in merged_titles.
        merged_titles = pd.concat([merged_titles, new_titles])

    # If current price is lower than next close price, a profit is positive.
    # Then a label is 1. Or the label is 0.
    # noinspection PyUnresolvedReferences
    merged_titles['label'] = (merged_titles['close'] < merged_titles['next_close']).astype(int)

    # Use only ['title', 'label'].
    profit_titles = merged_titles[['title', 'label']]
    profit_titles = profit_titles.dropna()
    return profit_titles


def calculate_volatility_titles():
    """

    :return volatility_titles: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
        column  title   | (str) The title.
                label   | (int) If current 5 minutes volatility is lower than next 5 minutes volatility,
                                a label is 1, else a label is 0.
    """
    stock_masters = get_stock_masters()
    merged_titles = pd.DataFrame()

    for i in tqdm(range(len(stock_masters)), desc='volatility_title'):
        stock_master = stock_masters.iloc[i:i + 1]
        naver_finance_forums = get_naver_finance_forums(stock_master).reset_index()
        stock_minute_prices = get_stock_minute_prices(stock_master).reset_index()

        # If there is no stock minute prices of this company, pass.
        if stock_minute_prices.empty:
            continue

        # Calculate a 5 minutes volatility and a next 5 minutes volatility.
        stock_minute_prices.loc[:, 'volatility'] = stock_minute_prices['close'].rolling(window=5).std()
        stock_minute_prices.loc[:, 'next_volatility'] = stock_minute_prices['volatility'].shift(-5)
        stock_minute_prices = stock_minute_prices.dropna()

        # Merge naver_finance_forums with stock_minute_prices
        new_titles = pd.merge(naver_finance_forums, stock_minute_prices, on=['code', 'date'], how='inner')
        new_titles = new_titles.set_index(['code', 'date', 'writer'])

        # Save new_titles in merged_titles.
        merged_titles = pd.concat([merged_titles, new_titles])

    # If current 5 minutes volatility is lower than next 5 minutes volatility,
    # then a label is 1. Or the label is 0.
    # noinspection PyUnresolvedReferences
    merged_titles['label'] = (merged_titles['volatility'] < merged_titles['next_volatility']).astype(int)

    # Use only ['title', 'label'].
    volatility_titles = merged_titles[['title', 'label']]
    volatility_titles = volatility_titles.dropna()
    return volatility_titles


def divide(dividend, divisor):
    quotient = dividend.copy()
    for dividend_column, divisor_column in zip(dividend.columns.values, divisor.columns.values):
        quotient[dividend_column] = dividend[dividend_column] / (divisor[divisor_column] + 1e-20)

    return quotient


# noinspection PyPep8Naming
def calculate_quantitative_behaviors():
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
    stock_masters = get_stock_masters()
    merged_quantitative_behaviors = pd.DataFrame()
    default_columns = ['adj_close', 'personal', 'national', 'investment', 'total_institution', 'other_finance',
                       'other_corporation',
                       'other_foreign', 'foreign', 'insurance', 'pef', 'pension', 'total_foreign', 'bank', 'trust',
                       'count']
    MA5_columns = ['{}_5'.format(column) for column in default_columns]
    MA20_columns = ['{}_20'.format(column) for column in default_columns]
    MA60_columns = ['{}_60'.format(column) for column in default_columns]
    MA1_MA5_columns = ['{}_1/5'.format(column) for column in default_columns]
    MA5_MA20_columns = ['{}_5/20'.format(column) for column in default_columns]
    MA20_MA60_columns = ['{}_20/60'.format(column) for column in default_columns]

    for i in tqdm(range(len(stock_masters)), desc='quantitative_behavior'):
        stock_master = stock_masters.iloc[i:i + 1]
        naver_finance_forum_stats = get_naver_finance_forum_stats(stock_master)
        stock_volumes = get_stock_volumes(stock_master)
        stock_prices = get_stock_prices(stock_master)

        # If a company is not covered, pass.
        if naver_finance_forum_stats.empty or stock_volumes.empty or stock_prices.empty:
            continue

        new_quantitative_behaviors = pd.concat([naver_finance_forum_stats, stock_volumes, stock_prices], axis=1)

        # new_quantitative_behaviors use only default columns.
        new_quantitative_behaviors = new_quantitative_behaviors[default_columns]

        # Set labels.
        new_quantitative_behaviors['next_close'] = new_quantitative_behaviors['adj_close'].shift(-1)
        # noinspection PyUnresolvedReferences
        new_quantitative_behaviors['label'] = (
                new_quantitative_behaviors['adj_close'] < new_quantitative_behaviors['next_close']).astype(int)

        # Calculate moving averages.
        new_quantitative_behaviors[MA5_columns] = new_quantitative_behaviors[default_columns].rolling(5).mean()
        new_quantitative_behaviors[MA20_columns] = new_quantitative_behaviors[default_columns].rolling(5).mean()
        new_quantitative_behaviors[MA60_columns] = new_quantitative_behaviors[default_columns].rolling(5).mean()
        new_quantitative_behaviors = new_quantitative_behaviors.dropna()

        # Calculate combinations of moving averages.
        # For preventing being divided by 0, add a small number.
        new_quantitative_behaviors[MA1_MA5_columns] = divide(new_quantitative_behaviors[default_columns],
                                                             new_quantitative_behaviors[MA5_columns])
        new_quantitative_behaviors[MA5_MA20_columns] = divide(new_quantitative_behaviors[MA5_columns],
                                                              new_quantitative_behaviors[MA20_columns])
        new_quantitative_behaviors[MA20_MA60_columns] = divide(new_quantitative_behaviors[MA20_columns],
                                                               new_quantitative_behaviors[MA60_columns])

        # merged_quantitative_behaviors use only MA1_MA5_columns, MA5_MA20_columns, MA20_MA60_columns.
        new_quantitative_behaviors = new_quantitative_behaviors[
            MA1_MA5_columns + MA5_MA20_columns + MA20_MA60_columns + ['label']]
        # Save new_titles in merged_titles.
        merged_quantitative_behaviors = pd.concat([merged_quantitative_behaviors, new_quantitative_behaviors])

    return merged_quantitative_behaviors


def calculate_high_frequency_volatilities():
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
    stock_masters = get_stock_masters()
    high_frequency_volatilities = pd.DataFrame()
    for i in tqdm(range(len(stock_masters)), desc='high_frequency_volatility'):
        stock_master = stock_masters.iloc[i:i + 1]
        stock_minute_prices = get_stock_minute_prices(stock_master)

        # Calculate 5 minutes volatility.
        stock_minute_prices.loc[:, 'volatility'] = stock_minute_prices['close'].rolling(window=5).std()

        # Make a 5 minutes later volatility a next_volatility and set it to label.
        stock_minute_prices.loc[:, 'label'] = stock_minute_prices['volatility'].shift(-1)

        # Drop first 4 rows and last 5 rows.
        stock_minute_prices = stock_minute_prices.dropna()

        # Merge calculated results of this company to high_frequency_volatilities.
        high_frequency_volatilities = pd.concat([high_frequency_volatilities, stock_minute_prices])

    return high_frequency_volatilities


def calculate_high_frequency_profits():
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
    stock_masters = get_stock_masters()
    high_frequency_profits = pd.DataFrame()
    for i in tqdm(range(len(stock_masters)), desc='high_frequency_profit'):
        stock_master = stock_masters.iloc[i:i + 1]
        stock_minute_prices = get_stock_minute_prices(stock_master)

        # Calculate 5 minutes volatility.
        stock_minute_prices.loc[:, 'volatility'] = stock_minute_prices['close'].rolling(window=5).std()

        # Make a 1 minute later close a next_close.
        stock_minute_prices.loc[:, 'next_close'] = stock_minute_prices['close'].shift(-1)
        # noinspection PyUnresolvedReferences
        stock_minute_prices.loc[:, 'label'] = (stock_minute_prices['close'] < stock_minute_prices['next_close']).astype(
            int)
        stock_minute_prices = stock_minute_prices.drop(columns=['next_close'])
        # noinspection PyUnresolvedReferences
        stock_minute_prices = stock_minute_prices.dropna()

        # Merge calculated results of this company to high_frequency_profits.
        high_frequency_profits = pd.concat([high_frequency_profits, stock_minute_prices])

    return high_frequency_profits


if __name__ == '__main__':
    # word_pack = calculate_word_pack()
    # word_pack.to_csv(DATA_DIR + 'word_pack.csv', index=False)

    # profit_titles = calculate_profit_titles()
    # profit_titles.to_csv(DATA_DIR + 'profit_titles.csv')

    # volatility_titles = calculate_volatility_titles()
    # volatility_titles.to_csv(DATA_DIR + 'volatility_title.csv')

    # quantitative_behaviors = calculate_quantitative_behaviors()
    # quantitative_behaviors.to_csv(DATA_DIR + 'quantitative_behavior.csv')

    stock_minute_volatilities = calculate_high_frequency_volatilities()
    stock_minute_volatilities.to_csv(DATA_DIR + 'high_frequency_volatility.csv')

    # stock_minute_volatilities = calculate_high_frequency_profits()
    # stock_minute_volatilities.to_csv(DATA_DIR + 'high_frequency_profit.csv')
