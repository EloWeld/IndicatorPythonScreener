import numpy as np
import pandas as pd
import ta


class RangeFilter:
    def __init__(self, close, per=100, mult=3.0):
        self.close = np.array(close)
        self.per = per
        self.mult = mult
        self.upColor = 'white'
        self.midColor = '#90bff9'
        self.downColor = 'blue'
        self.filt = np.zeros_like(self.close)
        self.upward = np.zeros_like(self.close)
        self.downward = np.zeros_like(self.close)
        self.hband = np.zeros_like(self.close)
        self.lband = np.zeros_like(self.close)
        self.filtcolor = np.array([''] * len(self.close))
        self.barcolor = np.array([''] * len(self.close))
        self.longCondition = np.zeros_like(self.close, dtype=bool)
        self.shortCondition = np.zeros_like(self.close, dtype=bool)
        
    def smoothrng(self, x, t, m):
        wper = 1
        avrng = ta.trend.EMAIndicator(close=pd.Series(np.abs(np.diff(x))), window=t).ema_indicator()
        smoothrng = ta.trend.EMAIndicator(close=avrng, window=wper).ema_indicator() * m
        smoothrng = np.concatenate(([0], smoothrng))  # Чтобы длины совпадали
        return smoothrng

    def rngfilt(self, x, r):
        rngfilt = np.copy(x)
        for i in range(1, len(x)):
            prev = rngfilt[i-1]
            if x[i] > prev:
                rngfilt[i] = prev if x[i] - r[i] < prev else x[i] - r[i]
            else:
                rngfilt[i] = prev if x[i] + r[i] > prev else x[i] + r[i]
        self.filt = rngfilt

    def calculate(self):
        smrng = self.smoothrng(self.close, self.per, self.mult)
        self.rngfilt(self.close, smrng)
        
        CondIni = np.zeros_like(self.close)  # Initialize CondIni

        
        for i in range(1, len(self.close)):
            self.upward[i] = self.upward[i-1] + 1 if self.filt[i] > self.filt[i-1] else 0
            self.downward[i] = self.downward[i-1] + 1 if self.filt[i] < self.filt[i-1] else 0
            
            self.hband[i] = self.filt[i] + smrng[i]
            self.lband[i] = self.filt[i] - smrng[i]
            
            longCond = (self.close[i] > self.filt[i] and self.close[i] > self.close[i-1] and self.upward[i] > 0) or \
                       (self.close[i] > self.filt[i] and self.close[i] < self.close[i-1] and self.upward[i] > 0)
            shortCond = (self.close[i] < self.filt[i] and self.close[i] < self.close[i-1] and self.downward[i] > 0) or \
                        (self.close[i] < self.filt[i] and self.close[i] > self.close[i-1] and self.downward[i] > 0)

            CondIni[i] = 1 if longCond else -1 if shortCond else CondIni[i-1]

            self.longCondition[i] = longCond and CondIni[i-1] == -1
            self.shortCondition[i] = shortCond and CondIni[i-1] == 1

        # Handle potential downcasting issue
        self.filtcolor = pd.Series(self.filtcolor).infer_objects(copy=False).to_numpy()
        self.barcolor = pd.Series(self.barcolor).infer_objects(copy=False).to_numpy()
