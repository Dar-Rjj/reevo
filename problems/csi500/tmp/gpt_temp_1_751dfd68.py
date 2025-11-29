import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Compute Raw Reversal-Momentum Signal
    # Intraday Reversal: (High - Close) / (Close - Low + 1e-12)
    intraday_reversal = (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-12)
    
    # Momentum Divergence: (Close - Open) / (High - Low + 1e-12)
    momentum_divergence = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-12)
    
    # Raw Reversal-Momentum Signal
    reversal_momentum_signal = intraday_reversal * momentum_divergence
    
    # Assess Volume Persistence
    # Volume Stability: Volume / 3-day Rolling Mean Volume
    volume_stability = data['volume'] / data['volume'].rolling(window=3, min_periods=1).mean()
    
    # Volume-Momentum Correlation: 8-day rolling correlation between momentum decay and volume
    momentum_decay = momentum_divergence.diff().fillna(0)
    
    # Calculate rolling correlation
    volume_momentum_corr = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 7:  # Need at least 8 days for correlation
            window_data = data.iloc[i-7:i+1]
            corr_val = window_data['volume'].corr(momentum_decay.iloc[i-7:i+1])
            volume_momentum_corr.iloc[i] = corr_val if not np.isnan(corr_val) else 0
        else:
            volume_momentum_corr.iloc[i] = 0
    
    # Volume Persistence Score
    volume_persistence = volume_stability * (1 + volume_momentum_corr.abs())
    
    # Generate Final Factor: Combine Reversal-Momentum × Volume Persistence
    final_factor = reversal_momentum_signal * volume_persistence
    
    return final_factor
