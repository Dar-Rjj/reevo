import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Adjusted Reversal Signal
    raw_reversal = (df['high'] - df['close']) / (df['close'] - df['low'])
    raw_reversal = raw_reversal.replace([np.inf, -np.inf], np.nan)
    
    volatility_expansion = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    volatility_expansion = volatility_expansion.replace([np.inf, -np.inf], np.nan)
    
    volatility_adjusted_reversal = raw_reversal * volatility_expansion
    
    # Momentum-Liquidity Pattern
    momentum_decay = (df['close'] - df['open']) - (df['open'] - df['close']).shift(1)
    
    volume_trend_acceleration = df['volume'] / df['volume'].shift(10)
    volume_trend_acceleration = volume_trend_acceleration.replace([np.inf, -np.inf], np.nan)
    
    # 8-day rolling correlation
    momentum_liquidity_corr = pd.Series(index=df.index, dtype=float)
    for i in range(7, len(df)):
        if i >= 8:
            window_momentum = momentum_decay.iloc[i-7:i+1]
            window_volume = volume_trend_acceleration.iloc[i-7:i+1]
            valid_mask = (~window_momentum.isna()) & (~window_volume.isna())
            if valid_mask.sum() >= 3:
                corr_val = window_momentum[valid_mask].corr(window_volume[valid_mask])
                momentum_liquidity_corr.iloc[i] = corr_val if not pd.isna(corr_val) else 0
            else:
                momentum_liquidity_corr.iloc[i] = 0
        else:
            momentum_liquidity_corr.iloc[i] = 0
    
    # Generate Confirmed Signal
    confirmed_signal = volatility_adjusted_reversal * momentum_liquidity_corr
    
    # Volume threshold filter
    volume_threshold = df['volume'].rolling(window=15, min_periods=1).quantile(0.3)
    volume_filter = df['volume'] > volume_threshold
    confirmed_signal = confirmed_signal * volume_filter
    
    # Apply intraday momentum efficiency sign
    intraday_momentum_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_momentum_efficiency = intraday_momentum_efficiency.replace([np.inf, -np.inf], np.nan)
    
    final_signal = confirmed_signal * np.sign(intraday_momentum_efficiency)
    
    return final_signal
