import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Reversal-Momentum Composite
    # Current Day Price Position
    data['dist_to_high'] = (data['high'] - data['close']) / data['high']
    data['dist_to_low'] = (data['close'] - data['low']) / data['low']
    
    # Intraday Momentum
    data['intraday_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['intraday_momentum'] = data['intraday_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Recent Extremes
    data['rolling_high_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['rolling_low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    
    # Extreme Proximity
    data['high_proximity'] = (data['rolling_high_3d'] - data['close']) / data['rolling_high_3d']
    data['low_proximity'] = (data['close'] - data['rolling_low_3d']) / data['rolling_low_3d']
    data['reversal_signal'] = np.minimum(data['high_proximity'], data['low_proximity'])
    
    # Combine Reversal and Momentum
    reversal_momentum = data['reversal_signal'] * data['intraday_momentum']
    
    # Apply Directional Weight
    directional_weight = np.where(data['low_proximity'] < data['high_proximity'], 1, -1)
    reversal_momentum_composite = reversal_momentum * directional_weight
    
    # Volatility Breakout Confirmation
    # Intraday Volatility Proxy
    data['range_volatility'] = (data['high'] - data['low']) / data['close']
    data['avg_range_volatility_20d'] = data['range_volatility'].rolling(window=20, min_periods=1).mean()
    data['volatility_ratio'] = data['range_volatility'] / data['avg_range_volatility_20d']
    
    # Apply Breakout Filter
    volatility_multiplier = np.where(data['volatility_ratio'] > 1.5, data['volatility_ratio'], 1.0)
    
    # Opening Momentum
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    gap_magnitude = np.abs(data['opening_gap'])
    
    # Combine with volatility breakout
    breakout_signal = reversal_momentum_composite * volatility_multiplier * gap_magnitude
    
    # Liquidity Efficiency Components
    # Trading Efficiency Metrics
    data['effective_spread'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    data['volume_to_amount'] = data['volume'] / data['amount']
    data['volume_to_amount'] = data['volume_to_amount'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Efficiency
    data['volume_amount_ema_5d'] = data['volume_to_amount'].ewm(span=5, min_periods=1).mean()
    data['volume_amount_max_20d'] = data['volume_to_amount'].rolling(window=20, min_periods=1).max()
    data['volume_amount_min_20d'] = data['volume_to_amount'].rolling(window=20, min_periods=1).min()
    data['volume_amount_range'] = data['volume_amount_max_20d'] - data['volume_amount_min_20d']
    data['volume_position'] = (data['volume_amount_ema_5d'] - data['volume_amount_min_20d']) / data['volume_amount_range']
    data['volume_position'] = data['volume_position'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # Market Efficiency Condition
    data['current_range'] = (data['high'] - data['low']) / data['close']
    data['avg_range_10d'] = data['current_range'].rolling(window=10, min_periods=1).mean()
    data['range_ratio'] = data['current_range'] / data['avg_range_10d']
    range_efficiency = 1 / data['range_ratio']
    
    # Volume Acceleration
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_acceleration'] = data['volume'] / data['prev_volume']
    data['volume_acceleration'] = data['volume_acceleration'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    efficiency_metric = range_efficiency * data['volume_acceleration']
    
    # Combine Liquidity Components
    trading_efficiency = data['effective_spread'] * data['volume_position']
    liquidity_composite = efficiency_metric * trading_efficiency
    
    # Volume Context
    data['volume_percentile_10d'] = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Final Alpha Factor
    final_signal = breakout_signal * liquidity_composite * data['volume_percentile_10d']
    
    # Return the factor values
    return final_signal
