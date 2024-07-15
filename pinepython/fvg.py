import numpy as np
import pandas as pd
import ta
from collections import deque


class FVGIndicator:
    def __init__(self, df, atr_multi=0.25):
        self.df = df
        self.atr_multi = atr_multi

    def run(self):
        atr = ta.volatility.average_true_range(self.df['high'], self.df['low'], self.df['close'], window=200) * self.atr_multi

        self.df['fvg_up'] = (self.df['low'] > self.df['high'].shift(2)) & (self.df['close'].shift(1) > self.df['high'].shift(2)) & (abs(self.df['low'] - self.df['high'].shift(2)) > atr)
        self.df['fvg_down'] = (self.df['high'] < self.df['low'].shift(2)) & (self.df['close'].shift(1) < self.df['low'].shift(2)) & (abs(self.df['low'].shift(2) - self.df['high']) > atr)

        self.df['InvFVG_BUY'] = 0
        self.df['InvFVG_SELL'] = 0

        for i in range(2, len(self.df)):
            if self.df['fvg_up'].iloc[i]:
                for j in range(i + 1, len(self.df)):
                    if self.df['low'].iloc[j] < self.df['high'].iloc[i - 2]:
                        self.df.loc[self.df.index[j], 'InvFVG_BUY'] = 1
                        break

            if self.df['fvg_down'].iloc[i]:
                for j in range(i + 1, len(self.df)):
                    if self.df['high'].iloc[j] > self.df['low'].iloc[i - 2]:
                        self.df.loc[self.df.index[j], 'InvFVG_SELL'] = 1
                        break

        return self.df
