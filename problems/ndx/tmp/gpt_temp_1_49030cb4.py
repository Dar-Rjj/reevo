import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Price Divergence Component
    # Morning Range Signal
    df['morning_range'] = (df['high'] - df['low']) / df['open']
    
    # Afternoon Deviation Signal
    df['afternoon_deviation'] = (df['close'] - (df['high'] + df['low'])/2) / df['open']
    
    # Combine price signals
    df['price_divergence'] = df['morning_range'] * df['afternoon_deviation']
    
    # Volume Adjustment Component
    # Volume Spike Detection (using rolling window with min_periods=1)
    df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_spike'] = df['volume'] / df['volume_ma5']
    
    # Volume Trend Strength
    df['volume_3days_ago'] = df['volume'].shift(3)
    df['volume_trend'] = df['volume'] / df['volume_3days_ago'].replace(0, 1)
    
    # Combine volume signals
    df['volume_adjustment'] = df['volume_spike'] * df['volume_trend']
    
    # Liquidity Adjustment Component
    # Calculate amount if not present (assuming amount = volume * typical price)
    if 'amount' not in df.columns:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['amount'] = df['volume'] * typical_price
    
    # Liquidity Score
    df['amount_ma10'] = df['amount'].rolling(window=10, min_periods=1).mean()
    df['liquidity_score'] = df['amount'] / df['amount_ma10']
    
    # Final Signal Synthesis
    df['raw_signal'] = df['price_divergence'] * df['volume_adjustment'] * df['liquidity_score']
    
    # Cross-sectional Z-score normalization
    def zscore(series):
        return (series - series.mean()) / series.std()
    
    df['factor'] = df.groupby(df.index)['raw_signal'].transform(zscore)
    
    return df['factor']
