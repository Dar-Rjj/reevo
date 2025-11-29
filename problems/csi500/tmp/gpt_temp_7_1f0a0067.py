import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Price Dynamics
    # High-Low Range Momentum
    data['high_5d_slope'] = data['high'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    data['low_5d_slope'] = data['low'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    data['high_low_momentum'] = (data['high_5d_slope'] + data['low_5d_slope']) / 2
    
    # Gap Momentum
    data['gap'] = data['close'] / data['open']
    data['gap_5d_slope'] = data['gap'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # 2. Volume Acceleration Patterns
    # Intraday Volume Dynamics
    data['volume_change_rate'] = data['volume'] / data['volume'].shift(1)
    data['volume_change_rate'] = data['volume_change_rate'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Volume range (using rolling high/low of volume)
    data['volume_high_5d'] = data['volume'].rolling(window=5).max()
    data['volume_low_5d'] = data['volume'].rolling(window=5).min()
    data['volume_range'] = data['volume_high_5d'] / data['volume_low_5d']
    data['volume_range'] = data['volume_range'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Multi-timeframe Volume Trends
    data['volume_5d_slope'] = data['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    data['volume_20d_slope'] = data['volume'].rolling(window=20).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # 3. Divergence-Convergence Signals
    # Price slopes for divergence calculations
    data['close_5d_slope'] = data['close'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    data['close_20d_slope'] = data['close'].rolling(window=20).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # Intraday Alignment
    data['price_volume_divergence'] = data['gap_5d_slope'] / data['volume_change_rate']
    data['price_volume_divergence'] = data['price_volume_divergence'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['range_volume_convergence'] = data['high_low_momentum'] * data['volume_5d_slope']
    
    # Multi-timeframe Dynamics
    data['short_term_divergence'] = data['close_5d_slope'] / data['volume_5d_slope']
    data['short_term_divergence'] = data['short_term_divergence'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['medium_term_convergence'] = data['close_20d_slope'] * data['volume_20d_slope']
    
    data['divergence_persistence'] = data['short_term_divergence'] / data['medium_term_convergence']
    data['divergence_persistence'] = data['divergence_persistence'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 4. Composite Signal Generation
    # Intraday Strength Assessment
    data['intraday_alignment'] = (data['price_volume_divergence'] + data['range_volume_convergence']) / 2
    data['gap_volume_reinforcement'] = data['gap_5d_slope'] * data['volume_5d_slope']
    data['intraday_strength'] = (data['intraday_alignment'] + data['gap_volume_reinforcement']) / 2
    
    # Multi-timeframe Validation
    data['consistency_score'] = np.sign(data['short_term_divergence']) * np.sign(data['medium_term_convergence'])
    data['signal_strength'] = data['intraday_strength'] * data['divergence_persistence']
    
    # Final composite factor
    data['factor'] = data['intraday_strength'] * data['signal_strength'] * data['consistency_score']
    
    # Clean infinite values and return
    factor_series = data['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
