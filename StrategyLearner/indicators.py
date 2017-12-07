"""Author: Liu Meihan"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data

def indicators(prices, lookback=10):
    """calculate four technical indicators"""
    prices.fillna(method = 'ffill', inplace = True)
    prices.fillna(method = 'bfill', inplace = True)
    # Momentum
    momentum = prices.copy()
    momentum.values[lookback:,] = prices.values[lookback:,]/prices.values[:-lookback,] - 1
    momentum.values[:lookback,] = np.nan
    # Price / SMA ratio
    sma = prices.rolling(window = lookback, min_periods = lookback).mean()
    sma = prices/sma
    # Bollinger Bands
    bbp = prices.copy()
    rolling_std = prices.rolling(window = lookback, min_periods = lookback).std()
    top_band = sma + rolling_std*2
    bottom_band = sma - rolling_std*2
    bbp = (prices - bottom_band)/(top_band - bottom_band)
    # Relative Strength Index
    daily_rets = prices.copy()
    daily_rets.values[1:,] = prices.values[1:,] - prices.values[:-1,]
    daily_rets.values[0] = np.nan
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1*daily_rets[daily_rets < 0].fillna(0).cumsum()
    up_gain = prices.copy()
    up_gain.ix[:] = 0
    up_gain.values[lookback:,] = up_rets.values[lookback:,] - up_rets.values[:-lookback,]
    down_loss = prices.copy()
    down_loss.ix[:] = 0
    down_loss.values[lookback:,] = down_rets.values[lookback:,] - down_rets.values[:-lookback,]
    rs = (up_gain/lookback) / (down_loss/lookback)
    rsi = 100 - 100/(1+rs)
    rsi.ix[:lookback,] = np.nan
    rsi[rsi == np.inf] = 100
    return momentum, sma, bbp, rsi

def test_code():
    """generator plots for the four indicators"""
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)
    prices = get_data(['JPM'], dates)
    prices.fillna(method = 'ffill', inplace = True)
    prices.fillna(method = 'bfill', inplace = True)
    prices.drop('SPY', axis = 1, inplace = True)
    prices = prices/prices.ix[0]
    momentum, sma, bbp, rsi = indicators(prices, lookback = 10)

    colname = prices.columns.values
    momentum = momentum.rename(columns = {colname[0]: 'Momentum'})
    ax1 = momentum.plot(color = 'k')
    ax1.set_title('Momentum', fontsize = 12)
    ax1.set_xlabel('Date', fontsize = 12)
    ax1.set_ylabel('Normalized Values', fontsize = 12)
    prices.plot(ax = ax1)

    sma = sma.rename(columns = {colname[0]: 'SMA'})
    ax2 = sma.plot(color = 'k')
    ax2.set_title('SMA', fontsize = 12)
    ax2.set_xlabel('Date', fontsize = 12)
    ax2.set_ylabel('Normalized Values', fontsize = 12)
    prices.plot(ax = ax2)

    bbp = bbp.rename(columns = {colname[0]: 'BBP'})
    ax3 = bbp.plot(color = 'k')
    df1 = bbp.copy()
    df1[:] = 0.7
    colname = bbp.columns.values
    df1 = df1.rename(columns = {colname[0]: 'top band'})
    df2 = bbp.copy()
    df2[:] = 0.15
    df2 = df2.rename(columns = {colname[0]: 'bottom band'})
    ax3.set_title('BBP', fontsize = 12)
    ax3.set_xlabel('Date', fontsize = 12)
    ax3.set_ylabel('Normalized Values', fontsize = 12)
    prices.plot(ax = ax3)
    df1.plot(ax = ax3)
    df2.plot(ax = ax3)
    plt.show()

    rsi = rsi.rename(columns = {colname[0]: 'RSI'})
    pricesc = prices.copy()
    pricesc = pricesc * 38.47
    ax4 = rsi.plot(color = 'k')
    colname = rsi.columns.values
    df1=rsi.copy()
    df1 = df1.rename(columns = {colname[0]: 'rsi=60'})
    df1[:] = 60
    df2 = rsi.copy()
    df2[:] = 30
    df2 = df2.rename(columns = {colname[0]: 'rsi=30'})
    ax4.set_title('RSI', fontsize = 12)
    ax4.set_xlabel('Date', fontsize = 12)
    ax4.set_ylabel('Normalized Values', fontsize = 12)
    pricesc.plot(ax = ax4)
    df1.plot(ax = ax4)
    df2.plot(ax = ax4)
    plt.show()

if __name__ == "__main__":
    test_code()