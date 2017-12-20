# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 3.
"""
from unittest import TestCase

from numpy import testing

from data_dealer.data_reader import *
# noinspection PyProtectedMember
from data_dealer.data_reader import _get_stock_master_table, _get_stock_price_table, _get_naver_finance_forum_table, \
    _get_naver_finance_forum_stat_table, _get_word_pack_table, _get_stock_minute_price_table, _get_stock_volume_table, \
    _get_profit_title_table, _get_quantitative_behavior_table, _get_volatility_title_table, \
    _get_high_frequency_volatility_table, _get_high_frequency_profit_table


class TestDataReader(TestCase):
    def test_get_stock_master_table(self):
        stock_masters = _get_stock_master_table()
        self.assertIsNotNone(stock_masters)
        self.assertEqual(2060, len(stock_masters))
        testing.assert_array_equal(['code'], stock_masters.index.names)
        testing.assert_array_equal(['name'], stock_masters.columns.values)

    def test_get_stock_price_table(self):
        stock_prices = _get_stock_price_table()
        self.assertIsNotNone(stock_prices)
        self.assertEqual(507006, len(stock_prices))
        testing.assert_array_equal(['code', 'date'], stock_prices.index.names)
        testing.assert_array_equal(['volume', 'open', 'high', 'low', 'close', 'market_capitalization',
                                    'listed_stocks_number', 'adj_close', 'adj_open'],
                                   stock_prices.columns.values)

    def test_get_stock_minute_price_table(self):
        stock_minute_prices = _get_stock_minute_price_table()
        self.assertIsNotNone(stock_minute_prices)
        self.assertEqual(65119728, len(stock_minute_prices))
        testing.assert_array_equal(['code', 'date'], stock_minute_prices.index.names)
        testing.assert_array_equal(['open', 'high', 'low', 'close', 'volume'],
                                   stock_minute_prices.columns.values)

    def test_get_stock_volume_table(self):
        stock_minute_prices = _get_stock_volume_table()
        self.assertIsNotNone(stock_minute_prices)
        self.assertEqual(476555, len(stock_minute_prices))
        testing.assert_array_equal(['code', 'date'], stock_minute_prices.index.names)
        testing.assert_array_equal(['personal', 'national', 'investment', 'total_institution',
                                    'other_finance', 'other_corporation', 'other_foreign', 'foreign',
                                    'insurance', 'pef', 'pension', 'total_foreign', 'bank', 'trust'],
                                   stock_minute_prices.columns.values)

    def test_get_naver_finance_forum_table(self):
        naver_finance_forums = _get_naver_finance_forum_table()
        self.assertIsNotNone(naver_finance_forums)
        self.assertEqual(6431219, len(naver_finance_forums))
        testing.assert_array_equal(['code', 'date', 'writer'], naver_finance_forums.index.names)
        testing.assert_array_equal(['title', 'opinion', 'hit', 'agreement', 'disagreement'],
                                   naver_finance_forums.columns.values)

    def test_get_naver_finance_forum_stat_table(self):
        naver_finance_forum_stats = _get_naver_finance_forum_stat_table()
        self.assertIsNotNone(naver_finance_forum_stats)
        self.assertEqual(502884, len(naver_finance_forum_stats))
        testing.assert_array_equal(['code', 'date'], naver_finance_forum_stats.index.names)
        testing.assert_array_equal(['count'], naver_finance_forum_stats.columns.values)

    def test_get_word_pack_table(self):
        word_pack = _get_word_pack_table()
        self.assertIsNotNone(word_pack)
        self.assertEqual(19733406, len(word_pack))
        testing.assert_array_equal(['code', 'date', 'word', 'writer'], word_pack.columns.values)

    def test_get_profit_title_table(self):
        profit_titles = _get_profit_title_table()
        self.assertIsNotNone(profit_titles)
        self.assertEqual(2272354, len(profit_titles))
        testing.assert_array_equal(['code', 'date', 'writer'], profit_titles.index.names)
        testing.assert_array_equal(['title', 'label'], profit_titles.columns.values)

    def test_get_volatility_title_table(self):
        volatility_titles = _get_volatility_title_table()
        self.assertIsNotNone(volatility_titles)
        self.assertEqual(1212757, len(volatility_titles))
        testing.assert_array_equal(['code', 'date', 'writer'], volatility_titles.index.names)
        testing.assert_array_equal(['title', 'label'], volatility_titles.columns.values)

    def test_get_quantitative_behavior_table(self):
        quantitative_behaviors = _get_quantitative_behavior_table()
        self.assertIsNotNone(quantitative_behaviors)
        self.assertEqual(362916, len(quantitative_behaviors))
        testing.assert_array_equal(['code', 'date'], quantitative_behaviors.index.names)
        testing.assert_array_equal(['adj_close_1/5', 'personal_1/5', 'national_1/5', 'investment_1/5',
                                    'total_institution_1/5', 'other_finance_1/5', 'other_corporation_1/5',
                                    'other_foreign_1/5', 'foreign_1/5', 'insurance_1/5', 'pef_1/5', 'pension_1/5',
                                    'total_foreign_1/5', 'bank_1/5', 'trust_1/5', 'count_1/5', 'adj_close_5/20',
                                    'personal_5/20', 'national_5/20', 'investment_5/20', 'total_institution_5/20',
                                    'other_finance_5/20', 'other_corporation_5/20', 'other_foreign_5/20',
                                    'foreign_5/20', 'insurance_5/20', 'pef_5/20', 'pension_5/20',
                                    'total_foreign_5/20', 'bank_5/20', 'trust_5/20', 'count_5/20',
                                    'adj_close_20/60', 'personal_20/60', 'national_20/60', 'investment_20/60',
                                    'total_institution_20/60', 'other_finance_20/60', 'other_corporation_20/60',
                                    'other_foreign_20/60', 'foreign_20/60', 'insurance_20/60', 'pef_20/60',
                                    'pension_20/60', 'total_foreign_20/60', 'bank_20/60', 'trust_20/60',
                                    'count_20/60', 'label'], quantitative_behaviors.columns.values)

    def test_get_high_frequency_volatility_table(self):
        high_frequency_volatilities = _get_high_frequency_volatility_table()
        self.assertIsNotNone(high_frequency_volatilities)
        self.assertEqual(46976733, len(high_frequency_volatilities))
        testing.assert_array_equal(['code', 'date'], high_frequency_volatilities.index.names)
        testing.assert_array_equal(['open', 'high', 'low', 'close', 'volume', 'profit', 'volatility', 'label'],
                                   high_frequency_volatilities.columns.values)

    def test_get_high_frequency_profit_table(self):
        high_frequency_profits = _get_high_frequency_profit_table()
        self.assertIsNotNone(high_frequency_profits)
        self.assertEqual(46977483, len(high_frequency_profits))
        testing.assert_array_equal(['code', 'date'], high_frequency_profits.index.names)
        testing.assert_array_equal(['open', 'high', 'low', 'close', 'volume', 'profit', 'volatility', 'label'],
                                   high_frequency_profits.columns.values)

    def test_get_stock_master(self):
        code = get_stock_masters().sample(1).index.values[0]
        stock_masters = get_stock_master(code)
        self.assertIsNotNone(stock_masters)
        self.assertEqual(1, len(stock_masters))
        self.assertEqual(code, stock_masters.index.values[0])

    def test_get_stock_masters(self):
        codes = get_stock_masters().sample(10).index.values
        stock_masters = get_stock_masters(codes)
        self.assertIsNotNone(stock_masters)
        self.assertEqual(10, len(stock_masters))
        testing.assert_array_equal(codes, stock_masters.index.values)

    def test_get_all_stock_masters(self):
        stock_masters = get_stock_masters()
        self.assertIsNotNone(stock_masters)
        self.assertEqual(2060, len(stock_masters))

    def test_get_stock_prices(self):
        stock_masters = get_stock_masters().sample(10)
        stock_prices = get_stock_prices(stock_masters)
        self.assertIsNotNone(stock_prices)
        self.assertEqual(2460, len(stock_prices))

    def test_get_naver_finance_forums(self):
        stock_masters = get_stock_master('000040')
        from_date = datetime(2016, 11, 1)
        to_date = datetime(2016, 11, 1)
        naver_finance_forums = get_naver_finance_forums(stock_masters, from_date, to_date)
        self.assertIsNotNone(naver_finance_forums)
        self.assertEqual(1, len(naver_finance_forums))

    def test_get_all_naver_finance_forums(self):
        naver_finance_forums = get_naver_finance_forums()
        self.assertIsNotNone(naver_finance_forums)
        self.assertEqual(6409727, len(naver_finance_forums))

    def test_get_naver_finance_forum_stats(self):
        stock_masters = get_stock_master('000040')
        from_date = datetime(2016, 11, 1)
        to_date = datetime(2016, 11, 1)
        naver_finance_forum_stats = get_naver_finance_forum_stats(stock_masters, from_date, to_date)
        self.assertIsNotNone(naver_finance_forum_stats)
        self.assertEqual(1, len(naver_finance_forum_stats))
        self.assertEqual(1, naver_finance_forum_stats.iloc[0]['count'])

    def test_get_naver_finance_forum_stats(self):
        naver_finance_forum_stats = get_naver_finance_forum_stats()
        self.assertIsNotNone(naver_finance_forum_stats)
        self.assertEqual(502884, len(naver_finance_forum_stats))
        self.assertEqual(1, naver_finance_forum_stats.iloc[0]['count'])
