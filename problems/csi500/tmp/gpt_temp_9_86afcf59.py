import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['price_deviation'] = (data['close'] - data['midpoint']) * data['volume']
    data['price_range'] = data['high'] - data['low']
    
    # Calculate midpoint momentum (avoid division by zero)
    mask = data['price_range'] > 0
    data['midpoint_momentum'] = np.where(mask, (data['close'] - data['open']) / data['price_range'], 0)
    
    # Calculate return extremes
    data['high_extreme'] = (data['high'] - data['close']) / data['close']
    data['low_extreme'] = (data['close'] - data['low']) / data['close']
    
    # Gap component
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = data['open'] - data['prev_close']
    data['gap_magnitude'] = data['opening_gap'].abs()
    
    # Volume divergence component
    data['recent_volume_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['recent_volume_avg']
    
    # Volume divergence flags
    data['low_volume_div'] = data['volume_ratio'] < 0.8
    data['high_volume_div'] = data['volume_ratio'] > 1.2
    
    # Identify extreme movement patterns using rolling quintiles
    data['high_extreme_rank'] = data['high_extreme'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['low_extreme_rank'] = data['low_extreme'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Extreme pattern flags
    data['high_extreme_pattern'] = data['high_extreme_rank'] > 0.8
    data['low_extreme_pattern'] = data['low_extreme_rank'] > 0.8
    
    # Generate combined signals
    data['reversal_signal'] = -data['price_deviation']
    data['momentum_signal'] = data['midpoint_momentum']
    
    # Enhance signals with gap magnitude when volume divergence present
    data['enhanced_reversal'] = data['reversal_signal'] * np.where(
        (data['high_extreme_pattern'] & data['low_volume_div']), 
        data['gap_magnitude'], 1
    )
    data['enhanced_momentum'] = data['momentum_signal'] * np.where(
        (data['low_extreme_pattern'] & data['high_volume_div']), 
        data['gap_magnitude'], 1
    )
    
    # Apply rolling standard deviation to combined signals
    data['reversal_std'] = data['enhanced_reversal'].rolling(window=5, min_periods=1).std()
    data['momentum_std'] = data['enhanced_momentum'].rolling(window=5, min_periods=1).std()
    
    # Normalize signals by their rolling standard deviation
    data['normalized_reversal'] = data['enhanced_reversal'] / (data['reversal_std'] + 1e-8)
    data['normalized_momentum'] = data['enhanced_momentum'] / (data['momentum_std'] + 1e-8)
    
    # Assign factor values based on patterns
    conditions = [
        # Strong Reversal: High Extreme with Volume Divergence
        data['high_extreme_pattern'] & data['low_volume_div'],
        # Strong Momentum: Low Extreme with Volume Divergence
        data['low_extreme_pattern'] & data['high_volume_div'],
        # Moderate Reversal: High Extreme without divergence
        data['high_extreme_pattern'] & ~data['low_volume_div'],
        # Moderate Momentum: Low Extreme without divergence
        data['low_extreme_pattern'] & ~data['high_volume_div']
    ]
    
    choices = [
        data['normalized_reversal'] * 1.5,  # Strong Reversal
        data['normalized_momentum'] * 1.5,  # Strong Momentum
        data['normalized_reversal'],         # Moderate Reversal
        data['normalized_momentum']          # Moderate Momentum
    ]
    
    data['factor'] = np.select(conditions, choices, default=0)  # Neutral for other cases
    
    # Final standardization for cross-sectional ranking
    factor_series = data['factor']
    
    return factor_series
