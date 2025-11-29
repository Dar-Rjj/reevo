import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling statistics for volume and price
    data['volume_median_5d'] = data['volume'].rolling(window=5, min_periods=3).median()
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_std_5d'] = data['volume'].rolling(window=5, min_periods=3).std()
    
    # Calculate returns
    data['ret_1d'] = data['close'] / data['close'].shift(1) - 1
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Calculate daily range
    data['daily_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['range_ma_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # Calculate VWAP
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['dollar_volume'] = data['typical_price'] * data['volume']
    data['cum_dollar_volume'] = data['dollar_volume'].rolling(window=5, min_periods=3).sum()
    data['cum_volume'] = data['volume'].rolling(window=5, min_periods=3).sum()
    data['vwap_5d'] = data['cum_dollar_volume'] / data['cum_volume']
    
    # Calculate gap pressure
    data['gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_abs'] = abs(data['gap'])
    
    # Calculate volume deviation
    data['volume_deviation'] = (data['volume'] - data['volume_median_5d']) / data['volume_median_5d']
    
    # Calculate momentum consistency
    data['momentum_alignment'] = np.sign(data['ret_1d']) * np.sign(data['ret_3d']) * np.sign(data['ret_5d'])
    data['momentum_strength'] = (abs(data['ret_1d']) + abs(data['ret_3d']) + abs(data['ret_5d'])) / 3
    
    # Calculate price-volume divergence
    data['pv_divergence'] = np.where(
        (data['momentum_strength'] > data['momentum_strength'].rolling(window=20, min_periods=10).quantile(0.7)) & 
        (abs(data['volume_deviation']) < 0.1),
        -1,  # Strong momentum, low volume → exhaustion
        np.where(
            (data['momentum_strength'] < data['momentum_strength'].rolling(window=20, min_periods=10).quantile(0.3)) & 
            (data['volume_deviation'] > 0.5),
            1,   # Weak momentum, high volume → accumulation
            0    # Normal alignment
        )
    )
    
    # Calculate intraday pressure
    data['price_vs_vwap'] = data['close'] / data['vwap_5d'] - 1
    data['pressure_score'] = data['gap_abs'] * np.sign(data['gap']) * np.where(
        data['volume'] > data['volume_median_5d'], 1, 0.5
    )
    
    # Calculate range expansion
    data['range_ratio'] = data['daily_range'] / data['range_ma_5d']
    data['range_expansion'] = np.where(
        (data['range_ratio'] > 1.5) & (data['volume_deviation'] > 0.3),
        data['range_ratio'] * data['volume_deviation'],
        0
    )
    
    # Calculate volume-weighted efficiency
    data['return_per_volume'] = abs(data['ret_1d']) / (data['volume'] + 1e-8)
    data['efficiency_ma_5d'] = data['return_per_volume'].rolling(window=5, min_periods=3).mean()
    data['efficiency_trend'] = data['return_per_volume'] / data['efficiency_ma_5d'] - 1
    
    # Calculate reversal signals
    data['vwap_deviation'] = data['close'] / data['vwap_5d'] - 1
    data['reversal_signal'] = np.where(
        (abs(data['vwap_deviation']) > 0.02) & (data['volume_deviation'] < -0.2),
        -np.sign(data['vwap_deviation']),  # Overextension with declining volume
        0
    )
    
    # Calculate session structure
    # Simplified: use morning vs afternoon momentum (assuming data has intraday structure)
    # For daily data, we'll use gap vs close relative performance
    data['session_structure'] = np.sign(data['gap']) * np.sign(data['ret_1d'])
    
    # Combine factors with weights
    for date in data.index:
        if pd.notna(data.loc[date, 'pv_divergence']) and pd.notna(data.loc[date, 'pressure_score']):
            # Cross-sectional ranking components (relative to recent history)
            pv_div = data.loc[date, 'pv_divergence']
            pressure = data.loc[date, 'pressure_score']
            expansion = data.loc[date, 'range_expansion']
            reversal = data.loc[date, 'reversal_signal']
            efficiency = data.loc[date, 'efficiency_trend']
            session = data.loc[date, 'session_structure']
            
            # Combine factors (equal weighting for simplicity)
            combined_score = (
                0.25 * pv_div +
                0.20 * pressure +
                0.15 * expansion +
                0.15 * reversal +
                0.15 * efficiency +
                0.10 * session
            )
            
            factor.loc[date] = combined_score
    
    # Fill NaN values with 0
    factor = factor.fillna(0)
    
    return factor
