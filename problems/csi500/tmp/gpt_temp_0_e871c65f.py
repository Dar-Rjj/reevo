import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume metrics
    df['prev_close'] = df['close'].shift(1)
    df['daily_range'] = (df['high'] - df['low']) / df['prev_close']
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Intraday Price-Volume Divergence Momentum
    # Assume first half = morning, second half = afternoon
    df['morning_high'] = df['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    df['morning_low'] = df['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    df['afternoon_high'] = df['high'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    df['afternoon_low'] = df['low'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    
    # Price extreme ratios
    df['morning_extreme_ratio'] = np.where(
        (df['open'] - df['morning_low']) != 0,
        (df['morning_high'] - df['open']) / (df['open'] - df['morning_low']),
        0
    )
    df['afternoon_extreme_ratio'] = np.where(
        (df['close'] - df['afternoon_low']) != 0,
        (df['afternoon_high'] - df['close']) / (df['close'] - df['afternoon_low']),
        0
    )
    
    # Volume concentration
    df['morning_volume'] = df['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    df['afternoon_volume'] = df['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    df['volume_ratio'] = np.where(
        df['afternoon_volume'] != 0,
        df['morning_volume'] / df['afternoon_volume'],
        1
    )
    
    # Divergence momentum signal
    price_momentum = np.where(
        df['morning_extreme_ratio'] * df['afternoon_extreme_ratio'] > 0,
        np.abs(df['morning_extreme_ratio'] - df['afternoon_extreme_ratio']),
        0
    )
    volume_confirmation = np.where(
        (df['volume_ratio'] > 1) & (df['morning_extreme_ratio'].abs() > df['afternoon_extreme_ratio'].abs()),
        1,
        np.where(
            (df['volume_ratio'] < 1) & (df['afternoon_extreme_ratio'].abs() > df['morning_extreme_ratio'].abs()),
            -1,
            0
        )
    )
    divergence_momentum = price_momentum * volume_confirmation
    
    # Amount-Weighted Gap Persistence
    df['gap_magnitude'] = np.abs(df['open'] - df['prev_close']) / df['prev_close']
    df['gap_filling'] = np.where(
        (df['open'] > df['prev_close']) & (df['low'] <= df['prev_close']),
        -1,  # Gap filled downward
        np.where(
            (df['open'] < df['prev_close']) & (df['high'] >= df['prev_close']),
            1,  # Gap filled upward
            0   # Gap not filled
        )
    )
    
    # Amount momentum and concentration
    df['amount_momentum'] = df['amount'].pct_change()
    amount_weighted_gap = df['gap_magnitude'] * (1 + df['amount_momentum'].fillna(0))
    gap_persistence = np.where(
        df['gap_filling'] == 0,
        amount_weighted_gap,  # Gap persists
        -amount_weighted_gap  # Gap filled
    )
    
    # Multi-Timeframe Price-Amount Convergence
    df['morning_return'] = (df['morning_high'] - df['open']) / df['open']
    df['afternoon_return'] = (df['close'] - df['open']) / df['open']
    
    # Amount flow consistency
    df['amount_flow'] = np.where(
        df['morning_return'] * df['afternoon_return'] > 0,
        np.abs(df['morning_return'] - df['afternoon_return']),
        0
    )
    
    # Price-amount alignment
    price_amount_alignment = np.where(
        (df['intraday_return'] * df['amount_momentum'].fillna(0)) > 0,
        df['intraday_return'].abs(),
        -df['intraday_return'].abs()
    )
    
    # Volume-Compression Breakout Efficiency
    df['range_compression'] = df['daily_range'].rolling(window=3, min_periods=1).std()
    df['volume_trend'] = df['volume'].pct_change().rolling(window=3, min_periods=1).mean()
    
    compression_efficiency = np.where(
        (df['range_compression'] < df['range_compression'].rolling(window=5, min_periods=1).mean()) &
        (df['volume_trend'] > 0),
        df['daily_range'] * (1 + df['volume_trend']),
        -df['daily_range']
    )
    
    # Cross-Session Momentum Transfer
    df['morning_volatility'] = (df['morning_high'] - df['morning_low']) / df['open']
    df['afternoon_volatility'] = (df['afternoon_high'] - df['afternoon_low']) / df['open']
    
    momentum_transfer = np.where(
        df['morning_return'] * df['afternoon_return'] > 0,
        (df['morning_return'].abs() + df['afternoon_return'].abs()) * 
        (1 - np.abs(df['morning_volatility'] - df['afternoon_volatility'])),
        -(df['morning_return'].abs() + df['afternoon_return'].abs())
    )
    
    # Combine all factors with weights
    weights = [0.25, 0.20, 0.20, 0.20, 0.15]  # Adjust based on importance
    factors = [
        divergence_momentum,
        gap_persistence,
        price_amount_alignment,
        compression_efficiency,
        momentum_transfer
    ]
    
    # Normalize and combine factors
    for i, factor in enumerate(factors):
        factor_series = pd.Series(factor, index=df.index)
        # Remove outliers and normalize
        factor_series = factor_series.clip(lower=factor_series.quantile(0.05), 
                                         upper=factor_series.quantile(0.95))
        if factor_series.std() != 0:
            factor_series = (factor_series - factor_series.mean()) / factor_series.std()
        factors[i] = factor_series
    
    # Weighted combination
    combined_factor = sum(weights[i] * factors[i] for i in range(len(weights)))
    
    result = combined_factor.fillna(0)
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'daily_range', 'intraday_return', 'morning_high', 
                   'morning_low', 'afternoon_high', 'afternoon_low', 'morning_extreme_ratio',
                   'afternoon_extreme_ratio', 'morning_volume', 'afternoon_volume', 
                   'volume_ratio', 'gap_magnitude', 'gap_filling', 'amount_momentum',
                   'morning_return', 'afternoon_return', 'amount_flow', 'range_compression',
                   'volume_trend', 'morning_volatility', 'afternoon_volatility']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return result
