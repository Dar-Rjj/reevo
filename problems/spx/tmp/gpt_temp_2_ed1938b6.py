import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Compute Momentum Component
    high_low_range = data['high'] - data['low']
    normalized_range = high_low_range / data['close']
    
    # Compute Volume Confirmation
    volume_change_rate = data['volume'] / data['volume'].shift(1)
    
    # Generate Volume Trend Signal
    volume_slope = data['volume'].rolling(window=3).apply(lambda x: (x[-1] - x[0]) / 3 if len(x) == 3 else 0)
    volume_signal = (volume_slope > 0).astype(int)
    
    # Generate Combined Factor
    momentum_weighted = normalized_range * volume_change_rate * volume_signal
    smoothed_momentum = momentum_weighted.ewm(alpha=0.8, adjust=False).mean()
    
    # Final smoothing
    factor = smoothed_momentum.rolling(window=3, min_periods=1).mean()
    
    return factor
