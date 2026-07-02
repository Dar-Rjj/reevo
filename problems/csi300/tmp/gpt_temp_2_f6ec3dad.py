import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency Signal
    normalized_price_range = ((df['high'] - df['low']) / df['open']) * 100
    closing_price_efficiency = ((df['close'] - df['open']) / (df['high'] - df['low'])) * 100
    price_efficiency_signal = normalized_price_range * closing_price_efficiency
    
    # Volume Efficiency Confirmation
    volume_concentration = df['volume'] / (df['high'] - df['low'])
    volume_concentration_zscore = volume_concentration.rolling(window=5).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std(), raw=False)
    volume_efficiency_confirmation = price_efficiency_signal * volume_concentration_zscore
    
    # Volatility Normalization
    daily_returns = df['close'].pct_change()
    rolling_volatility = daily_returns.rolling(window=10).std()
    
    # Combined Signal
    combined_signal = volume_efficiency_confirmation / rolling_volatility
    
    return combined_signal
