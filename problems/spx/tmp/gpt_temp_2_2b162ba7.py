import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Volume-Weighted Price Change
    df['vwpc'] = (df['close'] - df['open']) * df['volume']
    df['vol_ma_5'] = df['volume'].rolling(window=5).mean()
    df['liquidity_efficiency'] = df['vwpc'] / df['vol_ma_5']
    
    # Normalize by Price Efficiency
    df['price_efficiency'] = (df['high'] - df['low']) / df['close']
    df['normalized_efficiency'] = df['liquidity_efficiency'] / df['price_efficiency']
    
    # Compute Volume Trend Strength
    df['vol_slope_3'] = df['volume'].rolling(window=3).apply(lambda x: np.polyfit(range(3), x, 1)[0], raw=True)
    df['vol_slope_10'] = df['volume'].rolling(window=10).apply(lambda x: np.polyfit(range(10), x, 1)[0], raw=True)
    df['trend_strength'] = df['vol_slope_3'] / df['vol_slope_10']
    
    # Compare to Liquidity Efficiency
    df['divergence'] = np.abs(df['normalized_efficiency'] * df['trend_strength'])
    
    # Compute Volatility Ratio
    df['hl_range'] = df['high'] - df['low']
    df['hl_std_3'] = df['hl_range'].rolling(window=3).std()
    df['hl_std_10'] = df['hl_range'].rolling(window=10).std()
    df['volatility_ratio'] = df['hl_std_3'] / df['hl_std_10']
    
    # Scale Final Signal and Apply Sigmoid Normalization
    df['final_signal'] = df['divergence'] * df['volatility_ratio']
    df['final_signal_normalized'] = 1 / (1 + np.exp(-df['final_signal']))
    
    return df['final_signal_normalized']
