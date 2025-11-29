import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Volatility Efficiency Ratio
    df = df.copy()
    
    # Calculate True Range Components
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Compute Directional Movement Efficiency
    df['up_move'] = np.where(df['high'] > df['open'], df['high'] - df['open'], 0)
    df['down_move'] = np.where(df['low'] < df['open'], abs(df['low'] - df['open']), 0)
    df['directional_movement'] = df['up_move'] + df['down_move']
    df['efficiency_ratio'] = df['directional_movement'] / df['true_range']
    df['efficiency_ratio'] = df['efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Weighted Price Acceleration
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_2'] = df['close'].pct_change(2)
    df['price_acceleration'] = df['price_change_1'] - df['price_change_2'].shift(1)
    df['volume_weighted_acceleration'] = df['price_acceleration'] * df['volume'] / df['volume'].rolling(10).mean()
    
    # Opening Momentum Persistence
    df['first_hour_high'] = df['high'].rolling(2).max()  # Simplified proxy for first hour high
    df['first_hour_low'] = df['low'].rolling(2).min()    # Simplified proxy for first hour low
    df['first_hour_return'] = (df['first_hour_high'] - df['first_hour_low']) / df['open']
    df['daily_range'] = (df['high'] - df['low']) / df['open']
    df['persistence_ratio'] = df['first_hour_return'] / df['daily_range']
    df['persistence_ratio'] = df['persistence_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Amount-Based Volatility Clustering
    df['amount_volatility'] = df['amount'].rolling(5).std() / df['amount'].rolling(5).mean()
    df['amount_spike'] = df['amount'] / df['amount'].rolling(10).mean()
    df['price_change_amount'] = df['close'].pct_change(1)
    df['responsiveness_ratio'] = abs(df['price_change_amount']) / df['amount_spike']
    df['responsiveness_ratio'] = df['responsiveness_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Multi-Scale Range Breakout Probability
    df['short_range'] = (df['high'].rolling(2).max() - df['low'].rolling(2).min()) / df['close']
    df['medium_range'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
    df['long_range'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']
    df['range_compression'] = df['short_range'] / df['medium_range']
    df['range_nesting'] = df['medium_range'] / df['long_range']
    df['breakout_probability'] = (df['range_compression'] + df['range_nesting']) / 2
    
    # Price-Volume Convergence Divergence
    df['price_trend'] = df['close'].rolling(5).apply(lambda x: (x[-1] - x[0]) / x[0])
    df['volume_trend'] = df['volume'].rolling(5).apply(lambda x: (x[-1] - x[0]) / x[0])
    df['trend_alignment'] = np.sign(df['price_trend']) * np.sign(df['volume_trend'])
    df['convergence_strength'] = abs(df['price_trend']) * abs(df['volume_trend']) * df['trend_alignment']
    
    # Intraday Pressure Accumulation
    df['pressure_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['pressure_direction'] = np.where(df['close'] > df['open'], 1, -1)
    df['pressure_accumulation'] = df['pressure_direction'].rolling(3).sum() * df['pressure_position']
    
    # Combine all factors with equal weights
    factors = [
        'efficiency_ratio',
        'volume_weighted_acceleration', 
        'persistence_ratio',
        'responsiveness_ratio',
        'breakout_probability',
        'convergence_strength',
        'pressure_accumulation'
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        if factor in df.columns:
            normalized = (df[factor] - df[factor].rolling(20).mean()) / df[factor].rolling(20).std()
            combined_factor += normalized.fillna(0)
    
    return combined_factor
