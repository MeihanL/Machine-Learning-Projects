"""Author: Liu Meihan"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
import datetime as dt
from marketsimcode import compute_portvals
from indicators import indicators

def testPolicy(symbol, sd, ed, sv=100000):
    """implements trading strategy, returns a trades dataframe"""
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)
    prices.fillna(method = 'ffill', inplace = True)
    prices.fillna(method = 'bfill', inplace = True)
    prices.drop('SPY', axis = 1, inplace = True)
    prices = prices/prices.ix[0]
    momentum, sma, bbp, rsi = indicators(prices, lookback = 10)

    orders = prices.copy()
    orders[:] = np.nan
    sma_cross = pd.DataFrame(0, index = sma.index, columns = sma.columns)
    sma_cross[sma >= 1] = 1
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0

    orders[(rsi < 30) & (bbp < 0.15) & (momentum < -0.07)] = 1000
    orders[(rsi > 60) & (bbp > 0.7) & (momentum > 0.07)] = -1000
    orders[(rsi > 60) & (sma_cross == 1)] = 0
    orders.fillna(method = 'ffill', inplace = True)
    orders.fillna(0, inplace = True)
    orders[1:] = orders.diff()
    orders.ix[0] = 0
    return orders

def test_code():
    """test trading strategy and generate plot"""
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    orders = testPolicy('JPM', sd, ed)
    dates = pd.date_range(sd, ed)
    prices = get_data(['JPM'], dates)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices.drop('SPY', axis=1, inplace=True)
    prices = prices/prices.ix[0]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, portvals = compute_portvals(orders, prices)

    portvals = portvals/portvals.ix[0]
    bkorders = orders.copy()
    bkorders[:] = 0
    bkorders.ix[0] = 1000 #benchmark orders
    bkcum_ret, bkavg_daily_ret, bkstd_daily_ret, bksharpe_ratio, bkportvals = compute_portvals(bkorders, prices)
    bkportvals = bkportvals/bkportvals.ix[0]

    trades = orders.loc[(orders != 0).any(axis = 1)]
    holding = trades.cumsum()
    holding = holding.loc[(trades != 0).any(axis = 1)]
    short = holding.loc[(holding < 0).any(axis = 1)]
    long = holding.loc[(holding > 0).any(axis = 1)]

    fig = portvals.plot(color = 'k')
    bkportvals.plot(ax=fig, color = 'b')
    fig.set_title("Manual Strategy", fontsize = 20)
    fig.set_xlabel('Date', fontsize = 14)
    fig.set_ylabel('Portfolio Values', fontsize = 14)
    ymin, ymax = fig.get_ylim()
    fig.vlines(short.index, ymin, ymax, color = 'r')
    fig.vlines(long.index, ymin, ymax, color = 'g')
    plt.show()

if __name__ == "__main__":
    test_code()