# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 3.
"""
from unittest import TestCase

from numpy import testing

from data.data_reader import *
# noinspection PyProtectedMember
from data.data_reader import _get_stock_master_table, _get_stock_price_table, _get_naver_finance_forum_table, \
    _get_naver_finance_forum_stat_table


class TestDataReader(TestCase):
    def test_get_stock_master_table(self):
        stock_masters = _get_stock_master_table()
        self.assertIsNotNone(stock_masters)
        self.assertEqual(2061, len(stock_masters))
        testing.assert_array_equal(['code'], stock_masters.index.names)
        testing.assert_array_equal(['name'], stock_masters.columns.values)

    def test_get_stock_price_table(self):
        stock_prices = _get_stock_price_table()
        self.assertIsNotNone(stock_prices)
        self.assertEqual(len(stock_prices), 494175)
        testing.assert_array_equal(['code', 'date'], stock_prices.index.names)
        testing.assert_array_equal(['volume', 'open', 'high', 'low', 'close', 'market_capitalization',
                                    'listed_stocks_number', 'adj_close', 'adj_open'],
                                   stock_prices.columns.values)

    def test_get_naver_finance_forum_table(self):
        naver_finance_forums = _get_naver_finance_forum_table()
        self.assertIsNotNone(naver_finance_forums)
        self.assertEqual(6453164, len(naver_finance_forums))
        testing.assert_array_equal(['code', 'date', 'writer'], naver_finance_forums.index.names)
        testing.assert_array_equal(['title', 'opinion', 'hit', 'agreement', 'disagreement'],
                                   naver_finance_forums.columns.values)

    def test_get_naver_finance_forum_stat_table(self):
        naver_finance_forum_stats = _get_naver_finance_forum_stat_table()
        self.assertIsNotNone(naver_finance_forum_stats)
        self.assertEqual(752265, len(naver_finance_forum_stats))
        testing.assert_array_equal(['code', 'date'], naver_finance_forum_stats.index.names)
        testing.assert_array_equal(['count'], naver_finance_forum_stats.columns.values)

    def test_get_all_stock_masters(self):
        stock_masters = get_all_stock_masters()
        self.assertIsNotNone(stock_masters)
        self.assertEqual(2061, len(stock_masters))

    def test_get_stock_master(self):
        code = get_all_stock_masters().sample(1).index.values[0]
        stock_masters = get_stock_master(code)
        self.assertIsNotNone(stock_masters)
        self.assertEqual(1, len(stock_masters))
        self.assertEqual(code, stock_masters.index.values[0])

    def test_get_stock_masters(self):
        codes = get_all_stock_masters().sample(10).index.values
        stock_masters = get_stock_masters(codes)
        self.assertIsNotNone(stock_masters)
        self.assertEqual(10, len(stock_masters))
        testing.assert_array_equal(codes, stock_masters.index.values)

    def test_get_stock_prices(self):
        stock_masters = get_all_stock_masters().sample(10)
        stock_prices = get_stock_prices(stock_masters)
        self.assertIsNotNone(stock_prices)
        self.assertEqual(2440, len(stock_prices))

    def test_get_naver_finance_forums(self):
        stock_masters = get_stock_master('000040')
        from_date = datetime(2016, 10, 31)
        to_date = datetime(2016, 10, 31)
        naver_finance_forums = get_naver_finance_forums(stock_masters, from_date, to_date)
        self.assertIsNotNone(naver_finance_forums)
        self.assertEqual(2, len(naver_finance_forums))

    def test_get_naver_finance_forum_stats(self):
        stock_masters = get_stock_master('000040')
        from_date = datetime(2016, 10, 31)
        to_date = datetime(2016, 10, 31)
        naver_finance_forum_stats = get_naver_finance_forum_stats(stock_masters, from_date, to_date)
        self.assertIsNotNone(naver_finance_forum_stats)
        self.assertEqual(1, len(naver_finance_forum_stats))
        self.assertEqual(2, naver_finance_forum_stats.iloc[0]['count'])
