import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for volatility regime
    returns = df['close'].pct_change()
    
    # Volatility Regime Classification
    short_vol = returns.rolling(window=3).std().ewm(span=5).mean()
    long_vol = returns.rolling(window=10).std().ewm(span=10).mean()
    vol_ratio = short_vol / long_vol
    
    # Classify regimes
    regime = pd.Series(np.where(vol_ratio > 1.2, 'High', 
                               np.where(vol_ratio < 0.8, 'Low', 'Normal')), 
                      index=df.index)
    
    # Intraday Range Analysis
    morning_breakout = (df['high'] - df['open']) / df['open']
    afternoon_support = (df['close'] - df['low']) / df['close']
    range_strength = morning_breakout * afternoon_support
    
    # Volume-Confirmed Efficiency
    price_range_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'])
    price_range_efficiency = price_range_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Calculate volume threshold and filter
    vol_ma = df['volume'].rolling(window=5).mean()
    high_volume_mask = df['volume'] > 1.5 * vol_ma
    volume_confirmation = price_range_efficiency * df['volume'].where(high_volume_mask).rolling(window=5).mean()
    
    # Momentum Alignment
    price_momentum = df['close'] / df['close'].shift(3) - 1
    range_momentum = (df['high'] - df['low']) / (df['high'].shift(3) - df['low'].shift(3)) - 1
    range_momentum = range_momentum.replace([np.inf, -np.inf], np.nan)
    
    alignment = np.sign(price_momentum) * np.sign(range_momentum)
    
    # Support/Resistance Proximity
    distance_to_high = (df['high'].rolling(window=10).max() - df['close']) / df['close']
    distance_to_low = (df['close'] - df['low'].rolling(window=10).min()) / df['close']
    proximity = pd.concat([distance_to_high, distance_to_low], axis=1).min(axis=1)
    
    # Z-score normalization function
    def zscore_normalize(series):
        return (series - series.rolling(window=20).mean()) / series.rolling(window=20).std()
    
    # Normalize components
    range_strength_norm = zscore_normalize(range_strength)
    volume_confirmation_norm = zscore_normalize(volume_confirmation)
    alignment_norm = zscore_normalize(alignment)
    proximity_norm = zscore_normalize(proximity)
    
    # Adaptive Signal Synthesis with Regime-Based Weighting
    composite_factor = pd.Series(index=df.index, dtype=float)
    
    for idx in df.index:
        if regime.loc[idx] == 'High':
            # High Volatility weights: 40% Range, 35% Volume, 15% Momentum, 10% Proximity
            weighted_sum = (0.4 * range_strength_norm.loc[idx] + 
                          0.35 * volume_confirmation_norm.loc[idx] + 
                          0.15 * alignment_norm.loc[idx] + 
                          0.1 * proximity_norm.loc[idx])
        elif regime.loc[idx] == 'Low':
            # Low Volatility weights: 25% Range, 45% Volume, 20% Momentum, 10% Proximity
            weighted_sum = (0.25 * range_strength_norm.loc[idx] + 
                          0.45 * volume_confirmation_norm.loc[idx] + 
                          0.2 * alignment_norm.loc[idx] + 
                          0.1 * proximity_norm.loc[idx])
        else:
            # Normal regime: equal weighting
            weighted_sum = (0.25 * range_strength_norm.loc[idx] + 
                          0.25 * volume_confirmation_norm.loc[idx] + 
                          0.25 * alignment_norm.loc[idx] + 
                          0.25 * proximity_norm.loc[idx])
        
        composite_factor.loc[idx] = weighted_sum * range_strength.loc[idx] * alignment.loc[idx]
    
    return composite_factor
