import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # 1. Intraday Momentum Efficiency
    intraday_momentum = (data['close'] - data['open']) / data['open']
    volume_efficiency = intraday_momentum / data['volume']
    volatility_adjustment = intraday_momentum / (data['high'] - data['low'])
    intraday_efficiency = volume_efficiency * volatility_adjustment
    
    # 2. Gap Reversal Strength
    overnight_gap = (data['open'] - data['prev_close']) / data['prev_close']
    intraday_reversal = (data['close'] - data['open']) / (data['open'] - data['prev_close'])
    volume_confirmation = data['volume'] / data['prev_volume']
    
    # Handle division by zero in intraday_reversal
    intraday_reversal = intraday_reversal.replace([np.inf, -np.inf], np.nan)
    
    gap_reversal_strength = overnight_gap * intraday_reversal * volume_confirmation
    
    # 3. Price-Range Momentum Divergence
    range_momentum = (data['close'] - data['low']) / (data['high'] - data['low'])
    price_momentum = (data['close'] - data['prev_close']) / data['prev_close']
    divergence_signal = range_momentum - price_momentum
    
    # 4. Multi-Timeframe Volume-Price Alignment
    # Calculate rolling returns and volume changes
    data['return_5d'] = data['close'].pct_change(5)
    data['volume_change_5d'] = data['volume'].pct_change(5)
    data['return_20d'] = data['close'].pct_change(20)
    data['volume_change_20d'] = data['volume'].pct_change(20)
    
    short_term_alignment = data['return_5d'] / data['volume_change_5d']
    medium_term_alignment = data['return_20d'] / data['volume_change_20d']
    
    # Handle division by zero
    short_term_alignment = short_term_alignment.replace([np.inf, -np.inf], np.nan)
    medium_term_alignment = medium_term_alignment.replace([np.inf, -np.inf], np.nan)
    
    alignment_ratio = short_term_alignment / medium_term_alignment
    
    # Combine all factors using equal weighting
    # Normalize each component by cross-sectional z-score
    components = pd.DataFrame({
        'intraday_efficiency': intraday_efficiency,
        'gap_reversal_strength': gap_reversal_strength,
        'divergence_signal': divergence_signal,
        'alignment_ratio': alignment_ratio
    })
    
    # Calculate cross-sectional z-scores for each date
    normalized_components = components.groupby(components.index).transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Final factor is the average of normalized components
    factor_values = normalized_components.mean(axis=1)
    
    return factor_values
