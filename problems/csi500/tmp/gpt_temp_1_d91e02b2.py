import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate basic price components
    data['daily_range'] = data['high'] - data['low']
    data['close_position'] = (data['close'] - data['low']) / data['daily_range'].replace(0, np.nan)
    data['open_to_close'] = data['close'] - data['open']
    data['open_to_close_pct'] = data['open_to_close'] / data['open'].replace(0, np.nan)
    
    # Intraday Price Patterns
    # Normalized daily range (5-day rolling)
    data['norm_range'] = data['daily_range'] / data['close'].shift(1).replace(0, np.nan)
    data['range_ma_ratio'] = data['norm_range'] / data['norm_range'].rolling(window=5, min_periods=3).mean()
    
    # Close position strength (deviation from neutral 0.5)
    data['close_position_strength'] = (data['close_position'] - 0.5).abs()
    
    # Open-to-close momentum with sign consistency
    data['oc_momentum'] = data['open_to_close_pct'] * np.sign(data['open_to_close_pct'].rolling(window=3, min_periods=2).mean())
    
    # Volume Confirmation
    # Volume relative to recent average (10-day rolling)
    data['volume_ma'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma'].replace(0, np.nan)
    
    # Volume-price relationship (correlation over 5 days)
    def volume_price_corr(x):
        if len(x) < 3:
            return np.nan
        prices = x['close'].pct_change().dropna()
        volumes = x['volume'].pct_change().dropna()
        if len(prices) < 2 or len(volumes) < 2:
            return np.nan
        common_idx = prices.index.intersection(volumes.index)
        if len(common_idx) < 2:
            return np.nan
        return np.corrcoef(prices.loc[common_idx], volumes.loc[common_idx])[0, 1]
    
    # Calculate rolling volume-price correlation
    corr_values = []
    for i in range(len(data)):
        if i < 4:
            corr_values.append(np.nan)
            continue
        window_data = data.iloc[i-4:i+1]
        corr = volume_price_corr(window_data)
        corr_values.append(corr)
    
    data['volume_price_corr'] = corr_values
    
    # Combined factor calculation
    # Positive factors: range expansion, strong close position, consistent momentum, high volume confirmation
    data['intraday_momentum_factor'] = (
        data['range_ma_ratio'].fillna(1) *  # Range expansion
        (1 + data['close_position_strength'].fillna(0)) *  # Strong close position
        data['oc_momentum'].fillna(0) *  # Open-to-close momentum
        data['volume_ratio'].fillna(1) *  # Volume confirmation
        (1 + data['volume_price_corr'].fillna(0).abs())  # Volume-price relationship strength
    )
    
    # Final factor series
    factor = data['intraday_momentum_factor']
    
    return factor
