import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Range-Price Divergence Analysis
    # Range Efficiency Patterns
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_range'] = data['intraday_range'].replace(0, np.nan)  # Avoid division by zero
    
    # Compute intraday price efficiency: (Close - Open) / (High - Low)
    data['range_efficiency'] = (data['close'] - data['open']) / data['intraday_range']
    
    # Assess range-volume relationship: Volume / (High - Low)
    data['range_volume_ratio'] = data['volume'] / data['intraday_range']
    
    # Analyze Price-Volume Directional Alignment
    # Intraday price direction: Sign(Close - Open)
    data['price_direction'] = np.sign(data['close'] - data['open'])
    
    # Volume direction: Sign(Volume - Volume[1])
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Create directional divergence: Price direction × Volume direction
    data['directional_divergence'] = data['price_direction'] * data['volume_direction']
    
    # Multi-Timeframe Divergence Confirmation
    # Short-term Divergence (5-day)
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['short_term_divergence'] = data['price_momentum_5d'] * data['volume_momentum_5d']
    
    # Medium-term Divergence (20-day)
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    data['medium_term_divergence'] = data['price_momentum_20d'] * data['volume_momentum_20d']
    
    # Divergence Persistence and Regime Detection
    # Track Divergence Persistence
    # Directional persistence: Count same-sign divergence over 3 days
    data['directional_div_sign'] = np.sign(data['directional_divergence'])
    data['persistence_count'] = data['directional_div_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Magnitude persistence: Average range efficiency divergence over 3 days
    data['magnitude_persistence'] = data['range_efficiency'].rolling(window=3, min_periods=1).mean()
    
    # Detect Divergence Regime
    # Multi-timeframe alignment: Sign(short-term) × Sign(medium-term)
    data['timeframe_alignment'] = np.sign(data['short_term_divergence']) * np.sign(data['medium_term_divergence'])
    
    # Divergence trend: Short-term - Medium-term divergence
    data['divergence_trend'] = data['short_term_divergence'] - data['medium_term_divergence']
    
    # Regime quality: Timeframe alignment × Divergence trend
    data['regime_quality'] = data['timeframe_alignment'] * data['divergence_trend']
    
    # Final Composite Factor Construction
    # Core Divergence Integration
    # Intraday composite: Range efficiency × Directional divergence
    data['intraday_composite'] = data['range_efficiency'] * data['directional_divergence']
    
    # Multi-timeframe confirmation: Intraday × Short-term × Medium-term
    data['multi_timeframe_confirmation'] = (data['intraday_composite'] * 
                                          data['short_term_divergence'] * 
                                          data['medium_term_divergence'])
    
    # Enhanced Factor
    # Persistence weighting: Confirmed divergence × Persistence score
    persistence_score = data['persistence_count'] * data['magnitude_persistence']
    data['persistence_weighted'] = data['multi_timeframe_confirmation'] * persistence_score
    
    # Final factor: Persistence-weighted × Regime quality
    data['final_factor'] = data['persistence_weighted'] * data['regime_quality']
    
    # Return the final factor series
    return data['final_factor']
