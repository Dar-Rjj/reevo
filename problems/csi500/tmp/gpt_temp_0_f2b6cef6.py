import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Breakout Momentum with Volume Adjustment
    df['daily_range'] = df['high'] - df['low']
    df['prev_range'] = df['daily_range'].shift(1)
    df['breakout_ratio'] = df['daily_range'] / df['prev_range']
    df['breakout_ratio'] = df['breakout_ratio'].replace([np.inf, -np.inf], np.nan)
    
    df['volume_change'] = df['volume'] / df['volume'].shift(1)
    df['volume_change'] = df['volume_change'].replace([np.inf, -np.inf], np.nan)
    
    df['volume_price_consistency'] = np.where(
        (df['breakout_ratio'] > 1) & (df['volume_change'] > 1), 1,
        np.where((df['breakout_ratio'] < 1) & (df['volume_change'] < 1), 1, 0)
    )
    
    df['forward_return'] = df['close'].shift(-1) / df['close'] - 1
    df['momentum_signal'] = df['breakout_ratio'] * df['volume_price_consistency'] * df['forward_return']
    df['momentum_signal'] = df['momentum_signal'].rolling(window=5, min_periods=1).mean()
    
    # Opening Gap Persistence with Intraday Pressure
    df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_type'] = np.where(df['gap_size'] > 0, 1, -1)
    
    df['high_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    df['low_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    df['gap_remaining'] = np.where(
        df['gap_type'] == 1,
        (df['high'] - df['open']) / (df['high'] - df['low']),
        (df['open'] - df['low']) / (df['high'] - df['low'])
    )
    
    df['pressure_alignment'] = np.where(
        df['gap_type'] == 1,
        df['high_pressure'],
        df['low_pressure']
    )
    
    df['persistence_score'] = df['gap_size'] * df['pressure_alignment'] * df['gap_remaining']
    df['persistence_score'] = df['persistence_score'] * (df['amount'] / df['amount'].rolling(window=20, min_periods=1).mean())
    
    # Multi-Timeframe Volatility-Regulated Momentum
    df['short_momentum'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_strength'] = df['short_momentum'].abs()
    df['momentum_direction'] = np.sign(df['short_momentum'])
    
    df['volatility_5d'] = (df['high'] - df['low']).rolling(window=5, min_periods=1).std()
    df['close_volatility'] = df['close'].pct_change().rolling(window=10, min_periods=1).std()
    df['combined_volatility'] = (df['volatility_5d'] + df['close_volatility']) / 2
    
    df['regulated_signal'] = df['short_momentum'] / df['combined_volatility']
    df['regulated_signal'] = df['regulated_signal'] * (df['amount'] / df['amount'].rolling(window=20, min_periods=1).mean())
    
    # Price-Volume Divergence with Range Position
    df['price_change'] = df['close'].pct_change()
    df['volume_change_pct'] = df['volume'].pct_change()
    
    df['price_direction'] = np.sign(df['price_change'])
    df['volume_direction'] = np.sign(df['volume_change_pct'])
    
    df['divergence_basic'] = np.where(
        df['price_direction'] != df['volume_direction'],
        df['price_change'].abs() * df['volume_change_pct'].abs(),
        0
    )
    
    df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['range_weight'] = 1 - 2 * np.abs(df['range_position'] - 0.5)
    
    df['divergence_score'] = df['divergence_basic'] * df['range_weight']
    df['divergence_score'] = df['divergence_score'].rolling(window=5, min_periods=1).mean()
    
    # Amount-Weighted Breakout Probability
    df['dist_from_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    df['dist_from_low'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['range_position_sym'] = 1 - np.abs(df['range_position'] - 0.5) * 2
    
    df['amount_intensity'] = df['amount'] / df['amount'].rolling(window=20, min_periods=1).mean()
    
    df['breakout_probability'] = df['range_position_sym'] * df['amount_intensity']
    df['breakout_probability'] = df['breakout_probability'] * (1 + df['combined_volatility'] / df['combined_volatility'].rolling(window=20, min_periods=1).mean())
    
    # Intraday Session Momentum Carryover
    df['morning_high'] = df['high'].rolling(window=1, min_periods=1).apply(lambda x: x.max() if len(x) > 0 else np.nan)
    df['morning_low'] = df['low'].rolling(window=1, min_periods=1).apply(lambda x: x.min() if len(x) > 0 else np.nan)
    df['morning_momentum'] = (df['morning_high'] - df['open']) / df['open']
    
    df['afternoon_high'] = df['high'].rolling(window=1, min_periods=1).apply(lambda x: x.max() if len(x) > 0 else np.nan)
    df['afternoon_low'] = df['low'].rolling(window=1, min_periods=1).apply(lambda x: x.min() if len(x) > 0 else np.nan)
    df['afternoon_momentum'] = (df['close'] - df['afternoon_low']) / df['afternoon_low']
    
    df['session_alignment'] = np.where(
        np.sign(df['morning_momentum']) == np.sign(df['afternoon_momentum']),
        (df['morning_momentum'].abs() + df['afternoon_momentum'].abs()) / 2,
        0
    )
    
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
    df['carryover_score'] = df['session_alignment'] * df['volume_ratio']
    
    # Combine all factors
    factors = [
        'momentum_signal',
        'persistence_score', 
        'regulated_signal',
        'divergence_score',
        'breakout_probability',
        'carryover_score'
    ]
    
    # Normalize each factor
    for factor in factors:
        df[factor] = (df[factor] - df[factor].rolling(window=60, min_periods=1).mean()) / df[factor].rolling(window=60, min_periods=1).std()
    
    # Equal weighted combination
    combined_factor = df[factors].mean(axis=1)
    
    return combined_factor
