# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 13.
"""
import os

import pandas as pd

from data_dealer.data_reader import get_naver_finance_forums

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


if __name__ == '__main__':
    word_pack = calculate_word_pack()
    word_pack.to_csv(DATA_DIR + 'word_pack.csv', index=False)
