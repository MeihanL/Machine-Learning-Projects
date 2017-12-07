"""Author: Liu Meihan"""

import pandas as pd
import numpy as np

def compute_portvals(orders, prices, start_val=100000, commission=0, impact=0):
    """
    compute_portvals() implements trading orders and returns the portfolio values for each day

    orders, a datafram from which to read orders
    prices, adjusted prices of a stock
    start_val, starting value of the portfolio (initial cash available)
    commission, the fixed amount in dollars charged for each transaction (both entry and exit)
    impact, the amount the price moves against the trader compared to the historical data at each transaction
    """
    prices['Cash'] = pd.Series(np.ones(len(prices)), index = prices.index)
    Holdings = prices.copy()
    Holdings[:] = np.nan
    Holdings['Cash'] = pd.Series(np.zeros(len(Holdings)), index = Holdings.index)

    start_day = orders.index[0]
    Holdings.ix[start_day, 0] = orders.loc[start_day].values
    Holdings.ix[start_day, 'Cash'] = start_val - orders.loc[start_day].values * prices.ix[start_day, 0]
    for i in range(1, len(orders)):
        date = orders.index[i]
        preday = orders.index[i-1]
        Holdings.ix[date, 0] = Holdings.ix[preday, 0] + orders.loc[date].values
        if orders.loc[date].values < 0:
            Holdings.loc[date, 'Cash'] = Holdings.loc[preday, 'Cash']\
                                         - orders.loc[date].values*prices.ix[date, 0]*(1 - impact) - commission
        elif orders.loc[date].values > 0:
            Holdings.loc[date, 'Cash'] = Holdings.loc[preday, 'Cash']\
                                         - orders.loc[date].values*prices.ix[date, 0] * (1 + impact) - commission
        else:
            Holdings.loc[date, 'Cash'] = Holdings.loc[preday, 'Cash']
    Holdings.fillna(method = 'ffill', inplace = True)
    Holdings.fillna(method = 'bfill', inplace = True)

    portvals = (Holdings * prices).sum(axis = 1)
    cum_ret = portvals[-1]/portvals[0] - 1
    daily_rets = portvals/portvals.shift(1) - 1
    daily_rets = daily_rets[1:]
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(252.0) * avg_daily_ret/std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, portvals

def test_code():
    of = "./orders/orders-12-modified.csv"
    sv = 1000000
    # Process orders
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]
    else:
        "warning, code did not return a DataFrame"

    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()