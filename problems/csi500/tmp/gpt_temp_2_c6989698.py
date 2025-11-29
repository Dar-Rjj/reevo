import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Intraday Volatility
    # Intraday volatility using high and low prices
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['prev_intraday_range'] = data['intraday_range'].shift(1)
    
    # Normalized volatility ratio with logarithmic transformation
    data['vol_ratio'] = np.log1p(data['intraday_range'] / (data['prev_intraday_range'] + 1e-8))
    
    # 2. Construct Price Reversal Signal
    # Calculate distances from close to intraday extremes
    data['dist_to_high'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['dist_to_low'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Identify if close is near daily extreme (within 20% of range)
    data['near_high'] = (data['dist_to_high'] <= 0.2).astype(int)
    data['near_low'] = (data['dist_to_low'] <= 0.2).astype(int)
    
    # Volume confirmation - compare to 5-day average
    data['vol_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_surge'] = data['volume'] / (data['vol_ma_5'] + 1e-8)
    
    # Compute reversal probability
    data['reversal_signal'] = (
        (data['near_high'] * -1 + data['near_low'] * 1) *  # Direction: near high -> negative, near low -> positive
        np.minimum(data['volume_surge'], 3.0) *  # Cap volume surge effect
        (1 + data['vol_ratio'].abs())  # Volatility regime adjustment
    )
    
    # 3. Calculate Amount-Based Pressure
    # Daily turnover intensity
    data['avg_tx_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['tx_size_ma'] = data['avg_tx_size'].rolling(window=5, min_periods=1).mean()
    data['tx_size_ratio'] = data['avg_tx_size'] / (data['tx_size_ma'] + 1e-8)
    
    # Large transaction clustering - concentration ratio
    data['amount_ma_3'] = data['amount'].rolling(window=3, min_periods=1).mean()
    data['amount_concentration'] = data['amount'] / (data['amount_ma_3'] + 1e-8)
    
    # Directional pressure indicator
    price_change = data['close'].pct_change()
    data['pressure_divergence'] = (
        np.sign(price_change) * 
        data['amount_concentration'] * 
        np.where(data['tx_size_ratio'] > 1, 1, -0.5)  # Large transactions suggest institutional activity
    )
    
    # 4. Combine Signals with Time Decay
    # Exponential weighting for recent signals
    decay_factor = 0.9
    data['weighted_reversal'] = data['reversal_signal'].copy()
    data['weighted_pressure'] = data['pressure_divergence'].copy()
    
    # Apply exponential decay over 5 days
    for i in range(1, 6):
        data['weighted_reversal'] += data['reversal_signal'].shift(i) * (decay_factor ** i)
        data['weighted_pressure'] += data['pressure_divergence'].shift(i) * (decay_factor ** i)
    
    # Market regime adjustment using rolling volatility percentile
    vol_window = 20
    data['vol_percentile'] = data['intraday_range'].rolling(window=vol_window, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)) if len(x) == vol_window else 0
    )
    
    # Generate composite factor
    data['composite_factor'] = (
        data['weighted_reversal'] * 
        data['weighted_pressure'] * 
        (1 + 0.5 * data['vol_percentile'])  # Scale by volatility regime
    )
    
    # Final factor series
    factor = data['composite_factor'].fillna(0)
    
    return factor
