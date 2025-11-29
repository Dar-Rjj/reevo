import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Component
    intraday_vol = (df['high'] - df['low']) / df['open']
    
    # Volatility Trend
    vol_trend_5d = intraday_vol.rolling(window=5, min_periods=3).sum()
    vol_trend_10d = intraday_vol.rolling(window=10, min_periods=5).sum()
    vol_trend = vol_trend_5d - vol_trend_10d
    
    # Volatility Regime Signal
    vol_median_20d = intraday_vol.rolling(window=20, min_periods=10).median()
    vol_regime_signal = np.sign(vol_trend) * (intraday_vol / vol_median_20d)
    
    # Price-Volume Divergence Component
    price_momentum = (df['close'] - df['open']) * df['volume']
    volume_momentum = df['volume'] - df['volume'].rolling(window=5, min_periods=3).mean()
    
    # Divergence Signal
    def rolling_corr(x, y, window):
        return pd.Series([x.iloc[i-window+1:i+1].corr(y.iloc[i-window+1:i+1]) 
                         if i >= window-1 else np.nan for i in range(len(x))], index=x.index)
    
    divergence_corr = rolling_corr(price_momentum, volume_momentum, 3)
    divergence_signal = divergence_corr * (price_momentum - volume_momentum)
    
    # Regime Transition Detection
    high_low_range = df['high'] - df['low']
    open_close_range = (df['open'] - df['close']).abs()
    
    range_5d_mean = high_low_range.rolling(window=5, min_periods=3).mean()
    oc_range_5d_mean = open_close_range.rolling(window=5, min_periods=3).mean()
    
    volatility_breakout = (high_low_range / range_5d_mean) - (open_close_range / oc_range_5d_mean)
    
    # Combine components
    factor = vol_regime_signal * divergence_signal * volatility_breakout
    
    return factor
