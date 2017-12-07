""" Author: Liu Meihan
    Implementing a strategy learner based on Q-learning which is a reinforcement learning algorithm
    addEvidence() trains the Q learner on time series data to learn a trading strategy
    testPolicy() tests the learned strategy against new data and returns a trades dataframe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util as ut
import datetime as dt
import QLearner as ql
from indicators import indicators
from ManualStrategy import testPolicy
from marketsimcode import compute_portvals

class Trading(object):
    """
    symbol: the stock symbol to train on
    sd: A datetime object that represents the start date
    ed: A datetime object that represents the end date
    sv: Start value of the portfolio
    verbose: if False do not generate any output
    impact: The market impact of each transaction
    """
    class Action:
        LONG = 0
        SHORT = 1
        NOTHING = 2

    def __init__(self, symbol='IBM',
                 sd=dt.datetime(2008,1,1),
                 ed=dt.datetime(2009,12,31),
                 sv=10000,
                 verbose=False,
                 impact=0.0):
        self.symbol = symbol
        self.verbose = verbose
        self.impact = impact
        self.cash = sv
        self.portval = sv
        self.portvalafter = sv
        self.shares = 0
        self.sharesafter = 0
        self.diff = 0
        dates = pd.date_range(sd - dt.timedelta(100), ed)
        df = ut.get_data([symbol], dates)[symbol]
        normed_prices = df/df.iloc[0]
        daily_rets = df/df.shift(5) -1
        daily_rets.iloc[0:5] = 0

        momentum10, sma10, bbp10, rsi10 = indicators(normed_prices, lookback=10)
        df = pd.DataFrame(df).assign(dr=daily_rets).assign(momentum10=momentum10).\
                 assign(sma10=sma10).assign(bbp10=bbp10).assign(rsi10=rsi10)[sd:]
        self.df = df
        # convert real number into an integer
        self.momentum = pd.cut(self.df['momentum10'], 10, labels=False)
        self.sma = pd.cut(self.df['sma10'], 10, labels=False)
        self.bbp = pd.cut(self.df['bbp10'], 10, labels=False)
        self.rsi = pd.cut(self.df['rsi10'], 10, labels=False)
        self.market = df.iterrows()
        self.current = self.market.next()
        self.action = self.Action()

    def buy(self):
        self.sharesafter = 1000
        self.diff = self.sharesafter - self.shares
        price = self.current[1][self.symbol]
        holding = self.sharesafter * price
        self.cash -= self.diff * price * (1 + self.impact)
        self.portvalafter = self.cash + holding
        self.shares = self.sharesafter
        r = self.portvalafter - self.portval
        self.portval = self.portvalafter
        if self.diff == 0:
            return -1000
        elif self.diff != 0 and self.current[1]['dr'] > -0.03:
            return 0
        else:
            if self.verbose:
                print 'long 1000 shares at $%0.2f'.format(price)
            return r

    def sell(self):
        self.sharesafter = -1000
        self.diff = self.sharesafter - self.shares
        price = self.current[1][self.symbol]
        holding = self.sharesafter * price
        self.cash -= self.diff * price*(1 - self.impact)
        self.portvalafter = self.cash + holding
        self.shares = self.sharesafter
        r = self.portvalafter - self.portval
        self.portval = self.portvalafter
        if self.diff == 0:
            return -1000
        elif self.diff != 0 and self.current[1]['dr'] < 0.03:
            return 0
        else:
            if self.verbose:
                print 'short 1000 shares at $%0.2f'.format(price)
            return r

    def hold(self):
        price = self.current[1][self.symbol]
        holding = self.shares * price
        self.portvalafter = self.cash + holding
        r = self.portvalafter - self.portval
        self.portval = self.portvalafter
        if self.shares == 0:
            if self.verbose:
                print 'hold cash position'
            return 0
        elif self.shares == 1000:
            if self.verbose:
                print 'hold long position'
            return r
        elif self.shares == -1000:
            if self.verbose:
                print 'hold short position'
            return r

    def discretize(self):
        """combine intergers together into a single number
           which represent at once all the indicators of the stock on each day
        """
        date = self.current[0]
        s = 0
        s += 1000 * self.momentum[date] + 100*self.sma[date] + 10*self.bbp[date] + self.rsi[date]
        return int(s)

    def reward(self, action):
        """return how much money we get at the end of a trade action"""
        reward = {self.action.LONG: self.buy,
                  self.action.SHORT: self.sell,
                  self.action.NOTHING: self.hold}[action]()
        try:
            self.current = self.market.next()
            state = self.discretize()
        except StopIteration:
            return None, None
        return state, reward

    def state(self):
        price = self.current[1][self.symbol]
        return price * self.shares + self.cash

    def raw(self):
        return self.df

class StrategyLearner(object):

    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.ql = ql.QLearner(num_states=int(1e4),
                              num_actions=3,
                              alpha=0.1,
                              gamma=0.9,
                              rar=0.5,
                              radr=0.99,
                              dyna=0,
                              verbose=False)

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM",
                    sd=dt.datetime(2008,1,1),
                    ed=dt.datetime(2009,12,31),
                    sv=100000):
        cr = 0
        i = 0
        while i < 500:
            i += 1
            trade = Trading(symbol, sd, ed, sv, self.verbose)
            s = trade.discretize()
            a = self.ql.querysetstate(s)
            while True:
                s_prime, r = trade.reward(a)
                if s_prime is None:
                    break
                a = self.ql.query(s_prime, r)
            precr = cr
            cr = trade.state()
            if (cr == precr) & (i > 100):
                break

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2010,1,1),
                   ed=dt.datetime(2011,12,31),
                   sv=100000):
        trade = Trading(symbol, sd, ed, sv, self.verbose)
        df = trade.raw()
        s = trade.discretize()
        a = self.ql.querysetstate(s)
        actions = []
        actions.append(float(a))
        while True:
            s_prime, r = trade.reward(a)
            if s_prime is None:
                break
            a = self.ql.querysetstate(s_prime)
            actions.append(float(a))
        actions = pd.DataFrame(actions, index = df.index)
        actions[actions == 2.0] = np.nan
        actions[actions == 0.0] = 1000
        actions[actions == 1.0] = -1000
        actions.iloc[-1] = -1000
        actions.fillna(method = 'ffill', inplace = True)
        actions.fillna(0, inplace = True)
        orders = actions.copy()
        orders[1:] = orders.diff()
        return orders

def test_code():
    """compare Q learning strategy with manual strategy and generate plot"""
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)
    prices = ut.get_data(['JPM'], dates)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices.drop('SPY', axis=1, inplace=True)
    prices = prices / prices.ix[0]

    sl = StrategyLearner()
    sl.addEvidence('JPM')
    orders = sl.testPolicy('JPM', sd, ed)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, portvals = compute_portvals(orders, prices)
    portvals = portvals / portvals.ix[0]

    morders = testPolicy('JPM', sd, ed)
    mcum_ret, mavg_daily_ret, mstd_daily_ret, msharpe_ratio, mportvals = compute_portvals(morders, prices)
    mportvals = mportvals / mportvals.ix[0]

    bkorders = orders.copy()
    bkorders[:] = 0
    bkorders.ix[0] = 1000
    bkcum_ret, bkavg_daily_ret, bkstd_daily_ret, bksharpe_ratio, bkportvals = compute_portvals(bkorders, prices)
    bkportvals = bkportvals/bkportvals.ix[0]

    fig = portvals.plot(color = 'r')
    mportvals.plot(ax=fig, color = 'k')
    bkportvals.plot(ax=fig, color = 'b')
    fig.set_title("Comparing Qlearner with ManualS trategy", fontsize=20)
    fig.set_xlabel('Date', fontsize=14)
    fig.set_ylabel('Portfolio Values', fontsize=14)
    plt.show()

if __name__=="__main__":
    test_code()