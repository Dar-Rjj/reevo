import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Fracture with Volume-Price Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic intraday components
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Morning Momentum: (High - Open) / (Open - Low)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    data['morning_momentum'] = (data['high'] - data['open']) / (data['open'] - data['low'] + epsilon)
    
    # Afternoon Momentum: (Close - Mid) / (Mid - Open)
    data['afternoon_momentum'] = (data['close'] - data['mid_price']) / (data['mid_price'] - data['open'] + epsilon)
    
    # Momentum Fracture: Morning Momentum - Afternoon Momentum
    data['momentum_fracture'] = data['morning_momentum'] - data['afternoon_momentum']
    
    # Analyze Fracture Persistence (3-day rolling patterns)
    data['fracture_3d_ma'] = data['momentum_fracture'].rolling(window=3, min_periods=1).mean()
    data['fracture_3d_std'] = data['momentum_fracture'].rolling(window=3, min_periods=1).std()
    data['fracture_persistence'] = data['momentum_fracture'] / (data['fracture_3d_std'] + epsilon)
    
    # Momentum Regime Shifts
    data['momentum_change'] = data['momentum_fracture'].diff()
    data['regime_shift'] = data['momentum_change'].rolling(window=3, min_periods=1).apply(
        lambda x: 1 if (x > 0).sum() >= 2 else (-1 if (x < 0).sum() >= 2 else 0)
    )
    
    # Volume-Price Divergence Analysis
    # Calculate daily volume concentration (morning vs afternoon proxy)
    data['daily_range'] = data['high'] - data['low']
    data['price_change'] = data['close'] - data['open']
    
    # Volume concentration indicator
    data['volume_concentration'] = data['volume'] / (data['daily_range'] + epsilon)
    
    # Volume-Momentum Divergence
    data['volume_momentum_divergence'] = (data['volume_concentration'] - data['volume_concentration'].rolling(window=5, min_periods=1).mean()) * data['momentum_fracture']
    
    # Divergence persistence (5-day rolling)
    data['divergence_persistence'] = data['volume_momentum_divergence'].rolling(window=5, min_periods=1).apply(
        lambda x: (x > 0).sum() if x.iloc[-1] > 0 else -(x < 0).sum()
    )
    
    # Price Compression and Expansion Patterns
    data['intraday_range'] = data['high'] - data['low']
    data['range_5d_ma'] = data['intraday_range'].rolling(window=5, min_periods=1).mean()
    data['range_expansion'] = data['intraday_range'] / (data['range_5d_ma'] + epsilon)
    
    # Range-Momentum Relationship
    data['range_momentum_sync'] = data['range_expansion'] * np.sign(data['momentum_fracture'])
    
    # Breakout signals based on range expansion and momentum
    data['breakout_signal'] = ((data['range_expansion'] > 1.2) & (data['momentum_fracture'].abs() > data['momentum_fracture'].rolling(window=5, min_periods=1).mean())).astype(int)
    
    # Generate Momentum Fracture Prediction Factor
    # Combine components with appropriate weighting
    
    # Base factor: Momentum fracture weighted by volume divergence
    base_factor = data['momentum_fracture'] * data['volume_momentum_divergence']
    
    # Apply range expansion weighting
    range_weighted = base_factor * data['range_expansion']
    
    # Scale by divergence persistence strength
    persistence_scaled = range_weighted * (1 + data['divergence_persistence'] * 0.1)
    
    # Apply regime shift adjustment
    regime_adjusted = persistence_scaled * (1 + data['regime_shift'] * 0.05)
    
    # Apply breakout confirmation
    breakout_enhanced = regime_adjusted * (1 + data['breakout_signal'] * 0.1)
    
    # Final factor with multi-day pattern consideration
    data['momentum_fracture_factor'] = breakout_enhanced * (1 + data['fracture_persistence'] * 0.05)
    
    # Clean extreme values
    factor_series = data['momentum_fracture_factor'].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0 (neutral signal)
    factor_series = factor_series.fillna(0)
    
    # Return the factor series indexed by date
    return factor_series
