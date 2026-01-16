import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Intraday Range Normalization
    intraday_range = (df['high'] - df['low']) / df['close']
    
    # Price Efficiency Component
    price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Volume Trend Strength
    volume_ma = df['volume'].rolling(window=10).mean()
    volume_trend_strength = df['volume'] / volume_ma
    
    # Volume Efficiency Component
    volume_efficiency = df['volume'] / (df['high'] - df['low'])
    
    # Combined Signal
    combined_signal = price_efficiency * volume_efficiency
    
    # Rolling Volatility Adjustment
    rolling_volatility = intraday_range.rolling(window=10).std()
    
    # Directional Divergence Logic
    direction = df['close'] > df['open']
    divergence_signal = direction.apply(lambda x: -1 * combined_signal if x else combined_signal)
    
    # Volatility-Adjusted Divergence Signal
    volatility_adjusted_signal = divergence_signal / rolling_volatility
    
    return volatility_adjusted_signal
