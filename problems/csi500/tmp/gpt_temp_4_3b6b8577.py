import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Classification
    df['prev_close'] = df['close'].shift(1)
    df['daily_range'] = (df['high'] - df['low']) / df['prev_close']
    
    short_vol = df['daily_range'].rolling(window=3, min_periods=2).std()
    medium_vol = df['daily_range'].rolling(window=10, min_periods=5).std()
    vol_ratio = short_vol / medium_vol
    
    # High volatility regime threshold (vol_ratio > 1.2)
    high_vol_regime = vol_ratio > 1.2
    
    # Regime-Adaptive Breakout Component
    # For High Volatility: (High_2hour - Open)/Open × (Close - Low_2hour)/Close
    # Using first 2 hours approximation: high/low of first 120 minutes
    # Since we don't have intraday data, we'll use open-to-close range as proxy
    high_vol_breakout = ((df['high'] - df['open']) / df['open']) * ((df['close'] - df['low']) / df['close'])
    
    # For Low Volatility: ((High + Low)/2 - High.rolling(5).max())/High.rolling(5).max()
    low_vol_breakout = (((df['high'] + df['low']) / 2) - df['high'].rolling(window=5, min_periods=3).max()) / df['high'].rolling(window=5, min_periods=3).max()
    
    breakout_component = np.where(high_vol_regime, high_vol_breakout, low_vol_breakout)
    
    # Regime-Adaptive Reversal Component
    # For High Volatility: -sign((Close.shift(1) - Open.shift(1))/Open.shift(1)) × (Close - Open)/Open
    prev_return = (df['close'].shift(1) - df['open'].shift(1)) / df['open'].shift(1)
    high_vol_reversal = -np.sign(prev_return) * ((df['close'] - df['open']) / df['open'])
    
    # For Low Volatility: (High.rolling(10).max() - Close)/(High.rolling(10).max() - Low.rolling(10).min()) × True Range
    high_10max = df['high'].rolling(window=10, min_periods=5).max()
    low_10min = df['low'].rolling(window=10, min_periods=5).min()
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    low_vol_reversal = ((high_10max - df['close']) / (high_10max - low_10min)) * true_range
    
    reversal_component = np.where(high_vol_regime, high_vol_reversal, low_vol_reversal)
    
    # Volume-Liquidity Efficiency
    # Volume clustering: sum((Close - Open)/Volume) during high volume periods
    volume_quantile = df['volume'].rolling(window=20, min_periods=10).apply(lambda x: pd.Series(x).quantile(0.7), raw=True)
    high_volume_periods = df['volume'] > volume_quantile
    volume_clustering = ((df['close'] - df['open']) / df['volume']).where(high_volume_periods, 0).rolling(window=5, min_periods=3).sum()
    
    # Volume timing: correlation(Volume, (Close - Open)/Open) - correlation(Volume, (Close.shift(1) - Open.shift(1))/Open.shift(1))
    current_volume_corr = df['volume'].rolling(window=10, min_periods=5).corr((df['close'] - df['open']) / df['open'])
    prev_volume_corr = df['volume'].rolling(window=10, min_periods=5).corr((df['close'].shift(1) - df['open'].shift(1)) / df['open'].shift(1))
    volume_timing = current_volume_corr - prev_volume_corr
    
    # Liquidity efficiency: Amount/((High - Low) × Volume)
    liquidity_efficiency = df['amount'] / ((df['high'] - df['low']) * df['volume'])
    
    # Final Alpha Calculation
    base_signal = breakout_component + reversal_component
    volume_confirmation = volume_clustering * volume_timing * liquidity_efficiency
    final_alpha = base_signal * volume_confirmation
    
    return pd.Series(final_alpha, index=df.index)
