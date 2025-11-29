import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volume-Price Divergence
    # Intraday Divergence Structure
    data['intraday_divergence'] = (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['divergence_persistence'] = data['intraday_divergence'].rolling(window=3).mean()
    
    # Dual Divergence Acceleration
    data['short_term_div'] = data['intraday_divergence'].rolling(window=5).mean() - data['intraday_divergence'].rolling(window=10).mean()
    data['medium_term_div'] = data['intraday_divergence'].rolling(window=15).mean() - data['intraday_divergence'].rolling(window=30).mean()
    
    # Volume-Weighted Price Efficiency
    # Volume Persistence Integration
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['price_efficiency'] = abs(data['close'] - data['close'].shift(1)) * data['volume'] / data['amount'].replace(0, np.nan)
    data['volume_weighted_efficiency'] = data['price_efficiency'] * data['volume_ratio']
    
    # Volume Compression Analysis
    data['vol_ratio_3_1'] = data['volume'].rolling(window=3).mean() / data['volume'].rolling(window=1).mean()
    data['vol_ratio_7_3'] = data['volume'].rolling(window=7).mean() / data['volume'].rolling(window=3).mean()
    data['volume_compression'] = data['vol_ratio_3_1'] - data['vol_ratio_7_3']
    
    # Range Context Integration
    data['daily_range'] = data['high'] - data['low']
    data['range_percentile'] = data['daily_range'].rolling(window=20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5)
    data['range_weighted_efficiency'] = data['price_efficiency'] * data['range_percentile']
    
    # Range-Confirmed Opening Dynamics
    # Opening Range Analysis with Price Confirmation
    data['opening_range_util'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['abs_range_util'] = abs(data['opening_range_util'] - 0.5)
    
    # Range-Weighted Opening Efficiency
    data['price_movement'] = (data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    data['range_acceleration'] = (data['daily_range'] / data['daily_range'].shift(1).replace(0, np.nan)) - 1
    data['avg_range_10d'] = data['daily_range'].rolling(window=10).mean()
    data['range_vs_avg'] = data['daily_range'] / data['avg_range_10d'].replace(0, np.nan)
    data['range_weighted_momentum'] = data['price_movement'] * data['range_vs_avg']
    
    # Composite Divergence Integration
    # Volume-Persistent Divergence Core
    data['divergence_core'] = data['divergence_persistence'] * data['volume_ratio']
    data['dual_div_weighted'] = data['divergence_core'] * (data['short_term_div'] + data['medium_term_div'])
    
    # Volume Expansion Amplification
    data['volume_expansion'] = data['volume_ratio'] - 1
    data['abs_volume_expansion'] = abs(data['volume_expansion'])
    data['divergence_amplified'] = data['dual_div_weighted'] * data['abs_volume_expansion']
    
    # Range Breakout Confirmation
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    data['breakout_strength'] = data['daily_range'] / data['avg_range_5d'].replace(0, np.nan)
    data['breakout_confirmed_div'] = data['divergence_amplified'] * data['breakout_strength']
    
    # Final Factor Integration
    data['composite_factor'] = (
        data['breakout_confirmed_div'] * 0.4 +
        data['range_weighted_efficiency'] * 0.3 +
        data['range_weighted_momentum'] * 0.2 +
        data['volume_compression'] * 0.1
    )
    
    return data['composite_factor']
