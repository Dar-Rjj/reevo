import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_abs'] = np.abs(data['gap'])
    
    # Calculate intraday high/low for first half (morning session)
    # Assuming first half is first 4 hours of 8-hour trading day
    # We'll use rolling window to approximate morning session
    data['morning_high'] = data['high'].rolling(window=4, min_periods=1).max()
    data['morning_low'] = data['low'].rolling(window=4, min_periods=1).min()
    
    # Morning gap resolution efficiency
    data['morning_resolution'] = np.where(
        data['gap'] > 0,
        (data['morning_high'] - data['prev_close']) / (data['open'] - data['prev_close']),
        (data['prev_close'] - data['morning_low']) / (data['prev_close'] - data['open'])
    )
    data['morning_resolution'] = np.clip(data['morning_resolution'], 0, 2)
    
    # Full day gap resolution
    data['daily_resolution'] = np.where(
        data['gap'] > 0,
        (data['high'] - data['prev_close']) / (data['open'] - data['prev_close']),
        (data['prev_close'] - data['low']) / (data['prev_close'] - data['open'])
    )
    data['daily_resolution'] = np.clip(data['daily_resolution'], 0, 2)
    
    # Gap persistence (how much gap remains at close)
    data['gap_persistence'] = np.where(
        data['gap'] > 0,
        (data['close'] - data['prev_close']) / (data['open'] - data['prev_close']),
        (data['prev_close'] - data['close']) / (data['prev_close'] - data['open'])
    )
    
    # Volume-based gap confirmation
    data['gap_volume_ratio'] = data['gap_abs'] / (data['volume'] + 1e-8)
    data['volume_acceleration'] = data['volume'].pct_change(periods=2)
    
    # Amount concentration during gap resolution
    data['amount_ma'] = data['amount'].rolling(window=5, min_periods=1).mean()
    data['amount_concentration'] = data['amount'] / (data['amount_ma'] + 1e-8)
    
    # Range breakout metrics
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['morning_range'] = (data['morning_high'] - data['morning_low']) / data['open']
    data['range_efficiency'] = data['morning_range'] / (data['intraday_range'] + 1e-8)
    
    # Range compression/expansion cycles
    data['range_change'] = data['intraday_range'].pct_change()
    data['range_momentum'] = data['range_change'].rolling(window=3, min_periods=1).mean()
    
    # Volume-range coordination
    data['volume_range_corr'] = data['volume'].rolling(window=5, min_periods=1).corr(data['intraday_range'])
    
    # Sentiment persistence components
    data['morning_bias'] = (data['morning_high'] - data['open']) / (data['open'] - data['morning_low'] + 1e-8)
    data['afternoon_bias'] = (data['high'] - data['morning_high']) / (data['morning_low'] - data['low'] + 1e-8)
    
    # Volume acceleration during sessions
    data['morning_volume'] = data['volume'].rolling(window=4, min_periods=1).mean()
    data['afternoon_volume'] = data['volume'].rolling(window=4, min_periods=1).mean()
    data['volume_persistence'] = data['afternoon_volume'] / (data['morning_volume'] + 1e-8)
    
    # Calculate cross-sectional efficiency metrics (using rolling percentiles)
    data['resolution_rank'] = data['morning_resolution'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    data['volume_efficiency_rank'] = data['gap_volume_ratio'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Composite factor calculation
    # 1. Gap resolution efficiency component
    resolution_component = (
        data['morning_resolution'] * 0.4 + 
        data['daily_resolution'] * 0.3 + 
        data['resolution_rank'] * 0.3
    )
    
    # 2. Volume-amount confirmation component
    volume_component = (
        -data['gap_volume_ratio'].rank(pct=True) * 0.4 +  # Lower ratio is better
        data['volume_acceleration'].rank(pct=True) * 0.3 +
        data['amount_concentration'].rank(pct=True) * 0.3
    )
    
    # 3. Range breakout component
    range_component = (
        data['range_efficiency'].rank(pct=True) * 0.4 +
        data['range_momentum'].rank(pct=True) * 0.3 +
        data['volume_range_corr'].rank(pct=True) * 0.3
    )
    
    # 4. Sentiment persistence component
    sentiment_component = (
        data['morning_bias'].rank(pct=True) * 0.3 +
        data['afternoon_bias'].rank(pct=True) * 0.3 +
        data['volume_persistence'].rank(pct=True) * 0.4
    )
    
    # Final composite factor with directional bias from gap sign
    final_factor = (
        resolution_component * 0.35 +
        volume_component * 0.25 +
        range_component * 0.20 +
        sentiment_component * 0.20
    ) * np.sign(data['gap'])
    
    # Apply volume-amount validation filter
    volume_filter = np.where(
        (data['amount_concentration'] > data['amount_concentration'].rolling(window=20, min_periods=1).quantile(0.3)) &
        (data['volume_acceleration'] > data['volume_acceleration'].rolling(window=20, min_periods=1).quantile(0.3)),
        1.2,  # Amplify strong signals
        np.where(
            (data['amount_concentration'] < data['amount_concentration'].rolling(window=20, min_periods=1).quantile(0.2)) |
            (data['volume_acceleration'] < data['volume_acceleration'].rolling(window=20, min_periods=1).quantile(0.2)),
            0.8,  # Dampen weak signals
            1.0   # Neutral
        )
    )
    
    final_factor = final_factor * volume_filter
    
    # Remove any potential future data contamination
    final_factor = final_factor.shift(1)  # Ensure no lookahead bias
    
    return final_factor
