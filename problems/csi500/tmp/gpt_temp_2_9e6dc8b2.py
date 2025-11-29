import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price metrics
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = df['open'] - df['prev_close']
    df['gap_abs'] = np.abs(df['gap'])
    
    # Calculate intraday ranges and returns
    df['high_low_range'] = df['high'] - df['low']
    df['open_close_range'] = np.abs(df['close'] - df['open'])
    df['intraday_return'] = df['close'] - df['open']
    df['abs_intraday_return'] = np.abs(df['intraday_return'])
    
    # Calculate midday point (average of high and low)
    df['midday'] = (df['high'] + df['low']) / 2
    
    # Morning session metrics (open to midday)
    df['morning_return'] = df['midday'] - df['open']
    df['morning_return_abs'] = np.abs(df['morning_return'])
    
    # Afternoon session metrics (midday to close)
    df['afternoon_return'] = df['close'] - df['midday']
    df['afternoon_return_abs'] = np.abs(df['afternoon_return'])
    
    # Gap filling efficiency
    df['gap_filling'] = np.where(
        df['gap'] > 0,
        (df['low'] - df['prev_close']) / df['gap'],
        (df['high'] - df['prev_close']) / df['gap']
    )
    df['gap_filling'] = np.clip(df['gap_filling'], -1, 1)
    
    # Momentum divergence ratio
    df['momentum_divergence_ratio'] = np.where(
        df['morning_return_abs'] > 0,
        df['afternoon_return'] / df['morning_return'],
        0
    )
    
    # Intraday acceleration patterns
    df['range_acceleration'] = df['high_low_range'] / df['high_low_range'].shift(1)
    df['range_acceleration'] = df['range_acceleration'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Gap volatility intensity
    df['gap_vol_intensity'] = df['gap_abs'] / df['high_low_range'].rolling(window=5, min_periods=1).mean()
    df['gap_vol_intensity'] = df['gap_vol_intensity'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-price alignment (5-day rolling correlation)
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate rolling correlation between price and volume changes
    volume_price_corr = []
    for i in range(len(df)):
        if i < 4:
            volume_price_corr.append(0)
        else:
            window_prices = df['price_change'].iloc[i-4:i+1]
            window_volumes = df['volume_change'].iloc[i-4:i+1]
            if len(window_prices) >= 2 and len(window_volumes) >= 2:
                corr = np.corrcoef(window_prices, window_volumes)[0, 1]
                volume_price_corr.append(corr if not np.isnan(corr) else 0)
            else:
                volume_price_corr.append(0)
    df['volume_price_corr'] = volume_price_corr
    
    # Smart money flow patterns
    df['amount_per_volume'] = df['amount'] / df['volume']
    df['amount_per_volume'] = df['amount_per_volume'].replace([np.inf, -np.inf], 0).fillna(0)
    df['smart_money_ratio'] = df['amount_per_volume'] / df['amount_per_volume'].rolling(window=5, min_periods=1).mean()
    
    # Volume sustainability during gap periods
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Price efficiency calculations
    df['true_range'] = np.maximum(
        df['high_low_range'],
        np.maximum(
            np.abs(df['high'] - df['prev_close']),
            np.abs(df['low'] - df['prev_close'])
        )
    )
    df['efficiency_ratio'] = df['abs_intraday_return'] / df['true_range']
    df['efficiency_ratio'] = np.clip(df['efficiency_ratio'], 0, 1)
    df['efficiency_ratio'] = df['efficiency_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volatility regime identification
    df['intraday_volatility'] = df['high_low_range'] / df['close']
    df['volatility_regime'] = np.where(
        df['intraday_volatility'] > df['intraday_volatility'].rolling(window=10, min_periods=1).mean(),
        1,  # High volatility
        0   # Low volatility
    )
    
    # Multi-timeframe momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_divergence'] = df['momentum_5d'] - df['momentum_10d']
    
    # Range-momentum alignment
    df['range_momentum_alignment'] = df['efficiency_ratio'] * np.sign(df['momentum_5d'])
    
    # Composite factor components
    # Gap-momentum divergence component
    gap_momentum_div = df['gap'] * df['momentum_divergence_ratio'] * np.sign(df['intraday_return'])
    
    # Efficiency-adjusted component
    efficiency_adj = gap_momentum_div * df['efficiency_ratio']
    
    # Volume confirmation weight
    volume_weight = 1 + df['volume_price_corr'] * df['smart_money_ratio']
    
    # Volatility regime adjustment
    volatility_weight = np.where(
        df['volatility_regime'] == 1,
        1.2,  # High volatility premium
        0.8   # Low volatility discount
    )
    
    # Multi-timeframe confirmation
    timeframe_confirmation = 1 + np.abs(df['momentum_divergence']) * np.sign(df['range_momentum_alignment'])
    
    # Combine all components
    composite_factor = (
        efficiency_adj * 
        volume_weight * 
        volatility_weight * 
        timeframe_confirmation
    )
    
    # Apply non-linear transformation
    final_factor = np.tanh(composite_factor * 0.1)  # Scale for tanh
    
    # Return as Series with date index
    return pd.Series(final_factor, index=df.index)
