# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 13.
"""
import os

import pandas as pd
import numpy as np
from datetime import datetime
import progressbar

from data_dealer.data_reader import get_naver_finance_forums, get_stock_masters, get_stock_minute_prices, \
    get_stock_prices, get_naver_finance_forum_stats, get_stock_volumes

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
    for (code, date, writer), words in naver_finance_forums.iteritems():
        for word in words:
            word_pack.append({
                'code': code,
                'date': date,
                'writer': writer,
                'word': word,
            })

    word_pack = pd.DataFrame(word_pack)

    return word_pack


def calculate_titles():
    """

    :return titles: (DataFrame)
        index   code    | (str) 6 digits number string representing a company.
                date    | (datetime) The created date and time.
                writer  | (str) The writer of the forum.
        column  title   | (str) The title.
                label   | (int) If stock profit is positive, label is 1, else label is 0.
    """
    stock_masters = get_stock_masters()
    merged_titles = pd.DataFrame()

    # Initialize a progressbar.
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=(len(stock_masters) + 1)).start()
    for i in range(len(stock_masters)):
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

        # Update the progressbar.
        bar.update(i + 1)

    # Finish the progressbar.
    bar.finish()

    # If current price is lower than next close price, a profit is positive.
    # Then a label is 1. Or the label is 0.
    # noinspection PyUnresolvedReferences
    merged_titles['label'] = (merged_titles['close'] < merged_titles['next_close']).astype(int)

    # Use only ['title', 'label'].
    titles = merged_titles[['title', 'label']]
    titles = titles.dropna()
    return titles


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

    # Initialize a progressbar.
    widgets = [progressbar.Percentage(), progressbar.Bar()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=(len(stock_masters) + 1)).start()
    for i in range(len(stock_masters)):
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

        # Update the progressbar.
        bar.update(i + 1)

    # Finish the progressbar.
    bar.finish()

    return merged_quantitative_behaviors


if __name__ == '__main__':
    # word_pack = calculate_word_pack()
    # word_pack.to_csv(DATA_DIR + 'word_pack.csv', index=False)

    # titles = calculate_titles()
    # titles.to_csv(DATA_DIR + 'title.csv')

    quantitative_behaviors = calculate_quantitative_behaviors()
    quantitative_behaviors.to_csv(DATA_DIR + 'quantitative_behavior.csv')
