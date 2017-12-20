# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 20.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.arima_model import ARIMA

from data_dealer.data_reader import get_stock_master, get_high_frequency_volatilities


def ewma(stock_prices: pd.Series, window=1, lambda_window=30, lambda_percent=0.94):
    print('Start EWMA...')

    log_profit = np.log(stock_prices / stock_prices.shift(window))
    squared_log_profit = log_profit ** 2
    squared_log_profit = squared_log_profit.dropna()
    powers = np.arange(lambda_window)
    weights = (1 - lambda_percent) * (lambda_percent ** powers)

    smas = pd.np.convolve(squared_log_profit, weights, 'valid')
    return_series = pd.Series(smas, index=squared_log_profit.index[len(powers) - window:])

    print('EWMA is done!!')
    return return_series


def arima(history, targets, p, d, q, company_name=''):
    arima_predictions = list()
    for t in tqdm(range(len(targets)), desc=company_name + ' ARIMA'):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        arima_prediction = output[0][0]
        arima_predictions.append(arima_prediction)
        observed_value = targets[t]
        history.append(observed_value)

    return arima_predictions


if __name__ == '__main__':
    stock_masters = get_stock_master('005930')  # 삼성전자
    high_frequency_volatilities = get_high_frequency_volatilities(stock_masters)
    ewma_predictions = ewma(high_frequency_volatilities['close'])

    high_frequency_volatilities['label'].plot()
    ewma_predictions.plot()

    plt.show()
