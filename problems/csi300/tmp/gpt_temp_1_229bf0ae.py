import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Initialize output series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate required metrics
    df['gap'] = df['open'] / df['close'].shift(1) - 1
    df['abs_gap'] = df['gap'].abs()
    df['ma_volume'] = df['volume'].rolling(window=10, min_periods=1).mean()
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev_close = df.iloc[i-1]['close']
        
        # Gap Detection
        gap = current['gap']
        abs_gap = current['abs_gap']
        large_gap = abs_gap > 0.01
        
        # Liquidity Filter
        volume_condition = current['volume'] > 1.5 * current['ma_volume']
        
        # Price Confirmation Signal
        high_low_range = current['high'] - current['low']
        close_open_range = current['close'] - current['open']
        strong_confirmation = abs(close_open_range) > 0.5 * high_low_range
        
        # Calculate factor value
        if large_gap:
            if volume_condition:
                if strong_confirmation:
                    # Strong liquidity-adjusted reversal signal
                    factor_value = -gap * (current['volume'] / current['ma_volume'])
                else:
                    # Weak confirmation but still reversal signal
                    factor_value = -gap * 0.5 * (current['volume'] / current['ma_volume'])
            else:
                # Low liquidity, weaker signal
                factor_value = -gap * 0.2
        else:
            # Small gap, minimal signal
            factor_value = gap * 0.1
            
        factor_values.iloc[i] = factor_value
    
    return factor_values
