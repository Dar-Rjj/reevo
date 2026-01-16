import pandas as pd
def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate components
    for i in range(1, len(df)):
        current = df.iloc[i]
        past = df.iloc[:i]  # Only use past data
        
        # Momentum Component
        if (current['high'] - current['low']) > (current['close'] - current['open']):
            momentum = (current['high'] - current['low']) / current['close']
        else:
            momentum = (current['close'] - current['open']) / current['open']
        
        direction_confirmation = current['close'] > current['open']
        
        # Volume Adjustment
        if i >= 5:
            vol_ma_5 = past['volume'].iloc[-5:].mean()
            vol_ratio = current['volume'] / vol_ma_5
        else:
            vol_ratio = 1.0  # Default when not enough history
            
        if i >= 20:
            vol_ma_20 = past['volume'].iloc[-20:].mean()
            if current['volume'] > vol_ma_20:
                vol_scale = 1.2
            else:
                vol_scale = 0.8
        else:
            vol_scale = 1.0  # Default when not enough history
            
        volume_adjusted = vol_ratio * vol_scale
        
        # Volatility Adjustment
        if i >= 10:
            volatility_5 = past['close'].iloc[-5:].std()
            volatility_10 = past['close'].iloc[-10:].std()
            if volatility_5 > volatility_10:
                volatility_scale = 0.5
            else:
                volatility_scale = 1.0
        else:
            volatility_scale = 1.0  # Default when not enough history
            
        # Combine components
        raw_signal = momentum * direction_confirmation * volume_adjusted
        factor.iloc[i] = raw_signal * volatility_scale
    
    return factor
