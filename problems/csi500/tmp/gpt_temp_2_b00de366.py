import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining intraday efficiency, volume-volatility interaction, 
    and price-range dynamics signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns and ranges
    data['prev_close'] = data['close'].shift(1)
    data['daily_return'] = (data['close'] - data['prev_close']) / data['prev_close']
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    data['prev_range'] = data['daily_range'].shift(1)
    
    # 1. Intraday Price Efficiency Factors
    # Opening Efficiency Anomaly
    data['open_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['open_to_high'] = (data['high'] - data['open']) / data['open']
    data['open_to_low'] = (data['open'] - data['low']) / data['open']
    data['opening_efficiency'] = (data['open_to_high'] - data['open_to_low']) / (data['daily_range'] + 1e-8)
    
    # Midday Reversal Patterns
    data['midday_reversal'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['price_path_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # 2. Volume-Volatility Interaction Factors
    # Volume-Driven Volatility Compression
    data['volume_rank'] = data.groupby(data.index)['volume'].rank(pct=True)
    data['volatility_rank'] = data.groupby(data.index)['daily_range'].rank(pct=True)
    data['vol_vol_divergence'] = data['volume_rank'] - data['volatility_rank']
    
    # Volatility Expansion Momentum
    data['volatility_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['volatility_expansion'] = data['daily_range'] / (data['volatility_5d'] + 1e-8)
    data['vol_cluster_persistence'] = data['volatility_expansion'].rolling(window=3, min_periods=2).std()
    
    # 3. Price-Range Dynamics Factors
    # Range Exhaustion Signals
    data['range_contraction'] = data['daily_range'] / (data['prev_range'] + 1e-8)
    data['consecutive_contraction'] = (data['range_contraction'] < 1).rolling(window=3, min_periods=2).sum()
    
    # Relative Range Position Strength
    data['range_percentile'] = data.groupby(data.index)['daily_range'].rank(pct=True)
    data['range_momentum'] = data['range_percentile'] - data['range_percentile'].shift(1)
    
    # Calculate composite factor
    factors = [
        data['opening_efficiency'],
        -data['midday_reversal'],  # Negative for reversal
        data['price_path_efficiency'],
        -data['vol_vol_divergence'],  # Negative for compression signals
        data['volatility_expansion'],
        -data['vol_cluster_persistence'],  # Negative for clustering
        -data['consecutive_contraction'],  # Negative for exhaustion
        data['range_momentum']
    ]
    
    # Combine factors with equal weights
    factor_values = pd.concat(factors, axis=1).fillna(0)
    composite_factor = factor_values.mean(axis=1)
    
    # Final normalization
    final_factor = composite_factor.groupby(composite_factor.index).transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    return final_factor
