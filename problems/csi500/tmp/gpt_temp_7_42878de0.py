import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate Absolute Price Movement Efficiency
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['abs_price_move'] = abs(data['close'] - data['open'])
    data['price_efficiency'] = data['abs_price_move'] / (data['true_range'] + 1e-8)
    
    # Calculate Intraday Range Efficiency
    data['range_efficiency'] = (data['high'] - data['low']) / (data['amount'] + 1e-8)
    
    # Apply Momentum Decay Weighting for Range Efficiency
    data['range_eff_ewm'] = data['range_efficiency'].ewm(span=5, adjust=False).mean()
    data['range_eff_sma'] = data['range_efficiency'].rolling(window=10, min_periods=5).mean()
    data['range_momentum'] = data['range_eff_ewm'] - data['range_eff_sma']
    
    # Calculate Opening Gap Strength
    data['gap_pct'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    data['gap_direction'] = np.sign(data['gap_pct'])
    
    # Measure Intraday Range Utilization
    data['high_low_range'] = data['high'] - data['low']
    data['range_utilization'] = np.where(
        data['gap_direction'] > 0,
        (data['open'] - data['low']) / (data['high_low_range'] + 1e-8),
        (data['high'] - data['open']) / (data['high_low_range'] + 1e-8)
    )
    
    # Track consecutive same-direction gaps
    data['gap_dir_change'] = data['gap_direction'].diff().fillna(0)
    data['consecutive_gaps'] = 0
    for i in range(1, len(data)):
        if data['gap_dir_change'].iloc[i] == 0 and data['gap_direction'].iloc[i] != 0:
            data['consecutive_gaps'].iloc[i] = data['consecutive_gaps'].iloc[i-1] + 1
        elif data['gap_direction'].iloc[i] != 0:
            data['consecutive_gaps'].iloc[i] = 1
        else:
            data['consecutive_gaps'].iloc[i] = 0
    
    data['gap_persistence'] = data['range_utilization'] * data['consecutive_gaps']
    
    # Volume Persistence Analysis
    data['volume_change'] = data['volume'].pct_change()
    data['volume_direction'] = np.sign(data['volume_change'])
    
    # Rolling Window of Same-Sign Volume Days
    data['volume_dir_consistency'] = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            data['volume_dir_consistency'].iloc[i] = data['volume_dir_consistency'].iloc[i-1] + 1
        else:
            data['volume_dir_consistency'].iloc[i] = 1
    
    # Volume Acceleration Confirmation
    data['volume_3day_change'] = data['volume'].pct_change(periods=3)
    data['volume_10day_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_acceleration'] = data['volume_3day_change'] / (abs(data['volume'].pct_change()) + 1e-8)
    
    # Volume trend alignment with price momentum
    data['price_momentum'] = data['close'].pct_change(periods=3)
    data['volume_price_alignment'] = np.sign(data['volume_3day_change']) * np.sign(data['price_momentum'])
    
    # Generate Composite Alpha Signal
    data['momentum_efficiency_weighted'] = data['price_efficiency'] * data['volume_dir_consistency']
    data['scaled_momentum'] = data['momentum_efficiency_weighted'] * data['range_momentum']
    data['gap_enhanced'] = data['scaled_momentum'] * data['gap_persistence']
    data['volume_filtered'] = data['gap_enhanced'] * data['volume_acceleration'] * data['volume_price_alignment']
    
    # Final alpha factor
    alpha_factor = data['volume_filtered']
    
    return alpha_factor
