import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Momentum-Fracture Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components with proper shifting to avoid lookahead bias
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Handle NaN values from shifting
    data = data.fillna(method='bfill')
    
    # Fractured Momentum Analysis
    data['fractured_price_momentum'] = ((data['close'] - data['open']) / 
                                       (data['high'] - data['low'] + 1e-8) * 
                                       (data['volume'] / (data['prev_volume'] + 1e-8)))
    
    data['gap_momentum_persistence'] = ((data['open'] - data['prev_close']) / 
                                       (data['prev_high'] - data['prev_low'] + 1e-8) * 
                                       (data['amount'] / (data['prev_amount'] + 1e-8)))
    
    # Volume-Price Fracture Detection
    data['volume_fracture_intensity'] = (np.abs(data['volume'] / (data['prev_volume'] + 1e-8) - 
                                              data['amount'] / (data['prev_amount'] + 1e-8)) * 
                                       (data['high'] - data['low']))
    
    data['price_fracture_momentum'] = ((data['high'] - data['low']) / 
                                     (data['prev_high'] - data['prev_low'] + 1e-8) * 
                                     np.sign(data['close'] - data['open']))
    
    # Opening-Closing Fracture Patterns
    data['opening_fracture'] = ((data['open'] - data['low']) / 
                               (data['high'] - data['low'] + 1e-8) * 
                               (data['volume'] / (data['prev_volume'] + 1e-8)))
    
    data['closing_fracture'] = ((data['high'] - data['close']) / 
                               (data['high'] - data['low'] + 1e-8) * 
                               (data['amount'] / (data['prev_amount'] + 1e-8)))
    
    # Fracture Convergence Divergence
    data['fracture_convergence'] = data['opening_fracture'] - data['closing_fracture']
    data['volume_fracture_alignment'] = data['volume_fracture_intensity'] * data['price_fracture_momentum']
    
    # Multi-Timeframe Fracture Momentum
    data['short_term_fracture_momentum'] = data['fractured_price_momentum'] * data['volume_fracture_intensity']
    data['medium_term_fracture_persistence'] = data['gap_momentum_persistence'] * data['price_fracture_momentum']
    
    # Fracture-Based Alpha Construction
    data['primary_fracture'] = data['fracture_convergence'] * data['volume_fracture_alignment']
    data['momentum_fracture'] = data['short_term_fracture_momentum'] - data['medium_term_fracture_persistence']
    
    # Cross-sectional fracture ranking
    # Calculate cross-sectional ranks for each day
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    data['primary_fracture_rank'] = data.groupby(data.index)['primary_fracture'].transform(cross_sectional_rank)
    data['momentum_fracture_direction'] = np.sign(data['momentum_fracture'])
    
    # Final alpha factor: Rank adjusted by momentum direction
    data['fracture_alpha'] = data['primary_fracture_rank'] * data['momentum_fracture_direction']
    
    # Fracture Signal Validation - consistency checks
    data['volume_alignment_consistency'] = (data['volume_fracture_alignment'].rolling(window=5, min_periods=3).std() + 1e-8)
    data['momentum_persistence_check'] = (data['momentum_fracture'].rolling(window=5, min_periods=3).mean())
    
    # Apply validation adjustments
    validation_adjustment = np.where(
        (data['volume_alignment_consistency'] < data['volume_alignment_consistency'].quantile(0.8)) & 
        (np.abs(data['momentum_persistence_check']) > 0.1),
        1.0, 0.5
    )
    
    data['final_alpha'] = data['fracture_alpha'] * validation_adjustment
    
    return data['final_alpha']
