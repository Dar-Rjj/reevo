import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Momentum Analysis
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_direction'] = np.sign(data['opening_gap'])
    data['gap_strength'] = np.abs(data['opening_gap'])
    
    # Range Pattern Detection
    data['daily_range'] = data['high'] - data['low']
    data['range_ratio'] = data['daily_range'] / data['prev_close']
    
    # Range expansion streak
    data['range_increase'] = (data['daily_range'] > data['daily_range'].shift(1)).astype(int)
    data['range_expansion_streak'] = data['range_increase'].groupby(data.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if len(x) > 0 else 0, raw=False
    ).reset_index(level=0, drop=True)
    
    # Range contraction signal
    data['range_contraction'] = (data['daily_range'] < data['daily_range'].shift(1)).astype(int)
    
    # Intraday Momentum Pressure
    data['high_side_momentum'] = (data['high'] - data['close']) / (data['daily_range'] + 1e-8)
    data['low_side_momentum'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    data['opening_momentum'] = (data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    
    # Volume-Momentum Timing
    data['volume_ma'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_acceleration'] = data['volume'] / (data['volume_ma'] + 1e-8)
    
    # Volume during range phases
    data['expansion_volume'] = data['volume'] * data['range_increase']
    data['contraction_volume'] = data['volume'] * data['range_contraction']
    
    # Gap-momentum volume
    data['gap_volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['gap_momentum_volume'] = data['gap_volume_ratio'] * data['gap_strength']
    
    # Momentum Divergence
    data['gap_range_divergence'] = data['opening_gap'] - data['range_ratio']
    data['gap_intraday_divergence'] = data['opening_gap'] - data['opening_momentum']
    
    # Volatility Context
    data['range_volatility'] = data['daily_range'] / data['prev_close']
    data['volatility_regime'] = (data['range_volatility'] > data['range_volatility'].rolling(window=10, min_periods=1).mean()).astype(int)
    
    # Liquidity Efficiency
    data['turnover_efficiency'] = data['amount'] / (data['close'] * data['volume'] + 1e-8)
    
    # Composite Signal Generation
    # Pattern Strength Components
    data['range_score'] = data['range_expansion_streak'] * data['range_ratio']
    data['divergence_score'] = np.abs(data['gap_range_divergence']) + np.abs(data['gap_intraday_divergence'])
    data['volume_score'] = data['gap_momentum_volume'] * data['turnover_efficiency']
    data['momentum_score'] = data['opening_momentum'] * data['gap_strength']
    
    # Momentum Regime
    high_momentum = (data['opening_momentum'] > data['opening_momentum'].rolling(window=5, min_periods=1).mean())
    low_momentum = (data['opening_momentum'] < data['opening_momentum'].rolling(window=5, min_periods=1).mean())
    data['momentum_regime'] = high_momentum.astype(int) - low_momentum.astype(int)
    
    # Directional Signal
    data['upward_pressure'] = (data['low_side_momentum'] > data['high_side_momentum']).astype(int)
    data['downward_pressure'] = (data['high_side_momentum'] > data['low_side_momentum']).astype(int)
    
    # Volume confirmation
    data['volume_confirmation'] = (data['volume_acceleration'] > 1).astype(int)
    
    # Final Composite Factor
    data['composite_factor'] = (
        data['range_score'] * 0.2 +
        data['divergence_score'] * 0.25 +
        data['volume_score'] * 0.15 +
        data['momentum_score'] * 0.2 +
        data['momentum_regime'] * 0.1 +
        (data['upward_pressure'] - data['downward_pressure']) * data['volume_confirmation'] * 0.1
    )
    
    # Clean up intermediate columns
    result = data['composite_factor'].copy()
    
    return result
