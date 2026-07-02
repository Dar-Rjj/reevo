import pandas as pd
def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days for full calculations
            factor.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        past = df.iloc[:i]  # All data up to but not including current day
        
        # Short-term Reversal components
        if i >= 5:
            price_drop = (df.iloc[i-1]['close'] - df.iloc[i-5]['close']) / df.iloc[i-5]['close']
        else:
            price_drop = 0
            
        # Volume confirmation (5-day MA)
        if i >= 5:
            vol_ma = past.iloc[-5:]['volume'].mean()
            vol_confirmation = 1 if current['volume'] > vol_ma else 0
        else:
            vol_confirmation = 0
            
        reversal = price_drop * vol_confirmation
        
        # Volatility Normalization (20-day std)
        vol_20d = past.iloc[-20:]['close'].std()
        if vol_20d > 0:
            reversal_norm = reversal / vol_20d
        else:
            reversal_norm = 0
            
        # Intraday Strength
        intraday_strength = (current['close'] - current['open']) / current['open'] if current['open'] != 0 else 0
        
        # Combine signals
        factor.iloc[i] = reversal_norm * intraday_strength
        
    return factor
