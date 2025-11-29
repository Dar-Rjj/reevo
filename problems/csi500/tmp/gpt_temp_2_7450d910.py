import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic metrics
    data['range'] = data['high'] - data['low']
    data['open_gap'] = abs(data['open'] - data['close'].shift(1))
    data['close_open_diff'] = abs(data['close'] - data['open'])
    data['expansion_magnitude'] = data['close_open_diff'] / data['range'].replace(0, np.nan)
    
    # 20-day rolling calculations
    data['avg_range_20d'] = data['range'].rolling(window=20, min_periods=10).mean()
    data['avg_volume_20d'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['avg_amount_20d'] = data['amount'].rolling(window=20, min_periods=10).mean()
    data['avg_open_gap_20d'] = data['open_gap'].rolling(window=20, min_periods=10).mean()
    
    # Price Compression Detection
    data['range_ratio'] = data['range'] / data['avg_range_20d']
    data['range_percentile'] = data['range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    data['opening_range'] = data['open_gap'] / data['avg_open_gap_20d']
    data['compression_signal'] = (data['opening_range'] < 0.5).astype(int)
    
    data['volume_ratio'] = data['volume'] / data['avg_volume_20d']
    
    # Volume persistence (5-day correlation between volume and range)
    def volume_range_corr(x):
        if len(x) < 5:
            return 0
        vol_data = data.loc[x.index, 'volume']
        range_data = data.loc[x.index, 'range']
        return vol_data.corr(range_data)
    
    data['volume_persistence'] = data['range'].rolling(window=5, min_periods=3).apply(
        volume_range_corr, raw=False
    )
    
    # Expansion Breakout Mechanics
    data['expansion_duration'] = 0
    for i in range(1, len(data)):
        if data['expansion_magnitude'].iloc[i] > 0.6:
            data['expansion_duration'].iloc[i] = data['expansion_duration'].iloc[i-1] + 1
    
    data['amount_flow'] = data['amount'] / data['avg_amount_20d']
    data['price_volume_expansion'] = data['close_open_diff'] * data['volume_ratio']
    
    # Breakout Quality Assessment
    data['clean_breakout'] = ((data['close'] > data['high'].shift(1)) | 
                             (data['close'] < data['low'].shift(1))).astype(int)
    
    data['expansion_consistency'] = data['expansion_magnitude'].rolling(window=3, min_periods=2).mean()
    data['volume_confirmation'] = (data['volume_ratio'] > 1.2).astype(int)
    
    # Liquidity Regime Classification
    data['liquidity_regime'] = 'normal'
    data.loc[data['amount_flow'] > 1.5, 'liquidity_regime'] = 'high'
    data.loc[data['amount_flow'] < 0.8, 'liquidity_regime'] = 'low'
    
    # Market Microstructure Signals
    data['opening_gap_fill'] = data['close_open_diff'] / data['open_gap'].replace(0, np.nan)
    
    data['high_low_asymmetry'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    data['price_discovery_efficiency'] = data['close_open_diff'] / np.maximum(
        data['high'] - data['open'], data['open'] - data['low']
    ).replace(0, np.nan)
    
    # Calculate midday price (average of high and low)
    data['midday_price'] = (data['high'] + data['low']) / 2
    data['last_hour_movement'] = (data['close'] - data['midday_price']) / data['midday_price']
    
    # Composite Factor Assembly
    # Base compression-expansion factor
    data['compression_expansion_base'] = (1 - data['range_percentile']) * data['expansion_magnitude']
    
    # Liquidity regime multipliers
    regime_multipliers = {
        'high': data['amount_flow'] * (1 + data['volume_persistence'].fillna(0)),
        'normal': (data['expansion_magnitude'] + data['volume_ratio']) / 2 * 
                 (1 + data['expansion_duration'] / 10),
        'low': (1 / data['amount_flow'].clip(lower=0.1)) * 
               (data['expansion_magnitude'] > 0.7).astype(int)
    }
    
    data['regime_multiplier'] = np.select(
        [data['liquidity_regime'] == 'high', 
         data['liquidity_regime'] == 'normal',
         data['liquidity_regime'] == 'low'],
        [regime_multipliers['high'], 
         regime_multipliers['normal'],
         regime_multipliers['low']],
        default=1
    )
    
    # Duration weighting
    data['duration_weight'] = 1 + (data['expansion_duration'] * 0.1)
    
    # Final factor calculation
    data['factor'] = (data['compression_expansion_base'] * 
                     data['regime_multiplier'] * 
                     data['price_discovery_efficiency'].fillna(0) * 
                     data['duration_weight'])
    
    # Apply signal refinement filters
    data['final_factor'] = data['factor'] * data['clean_breakout'] * data['volume_confirmation']
    
    # Return the factor series
    return data['final_factor']
