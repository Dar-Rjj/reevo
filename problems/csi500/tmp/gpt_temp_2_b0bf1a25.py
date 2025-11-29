import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining price momentum, volume interactions, and volatility patterns
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Price-Based Momentum Factors
    # High-Low Range Momentum
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['prev_range'] = data['daily_range'].shift(1)
    data['range_momentum'] = data['daily_range'] / (data['prev_range'] + 1e-8)
    data['range_expansion'] = np.where(data['daily_range'] > data['prev_range'], 1, -1)
    
    # Opening Gap Efficiency
    data['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['gap_filling'] = np.where(
        (data['gap_size'] > 0) & (data['low'] <= data['close'].shift(1)), -1,
        np.where((data['gap_size'] < 0) & (data['high'] >= data['close'].shift(1)), -1, 1)
    )
    
    # Close-to-Close Momentum
    data['ret_1d'] = data['close'].pct_change()
    data['ret_3d'] = data['close'].pct_change(3)
    data['momentum_persistence'] = np.sign(data['ret_1d']) * np.sign(data['ret_3d'])
    data['momentum_magnitude'] = data['ret_1d'].abs() / (data['ret_3d'].abs().rolling(5).mean() + 1e-8)
    
    # Volume-Price Interaction Factors
    # Amount-Price Impact
    data['amount_ma'] = data['amount'].rolling(5).mean()
    data['high_amount_momentum'] = np.where(
        data['amount'] > data['amount_ma'],
        data['ret_1d'] * (data['amount'] / data['amount_ma']),
        data['ret_1d'] * (data['amount_ma'] / (data['amount'] + 1e-8))
    )
    
    # Volume-Momentum Alignment
    data['volume_ma'] = data['volume'].rolling(5).mean()
    data['volume_confirmation'] = np.sign(data['ret_1d']) * np.where(
        data['volume'] > data['volume_ma'], 1, -1
    )
    data['volume_divergence'] = np.where(
        (data['ret_1d'] > 0) & (data['volume'] < data['volume_ma']), -1,
        np.where((data['ret_1d'] < 0) & (data['volume'] > data['volume_ma']), -1, 1)
    )
    
    # Volume Concentration Timing
    data['open_volume_ratio'] = data['volume'].rolling(3).apply(
        lambda x: x.iloc[0] / (x.sum() + 1e-8) if len(x) == 3 else np.nan
    )
    data['volume_distribution'] = data['volume'].rolling(5).std() / (data['volume'].rolling(5).mean() + 1e-8)
    
    # Volatility-Momentum Integration
    # Range-Based Volatility Clustering
    data['volatility_5d'] = data['daily_range'].rolling(5).std()
    data['high_vol_momentum'] = np.where(
        data['daily_range'] > data['volatility_5d'],
        data['ret_1d'] * (data['daily_range'] / data['volatility_5d']),
        data['ret_1d'] * (data['volatility_5d'] / (data['daily_range'] + 1e-8))
    )
    
    # Intraday Momentum Evolution
    data['early_momentum'] = (data['high'].rolling(3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == 3 else np.nan
    ))
    data['late_momentum'] = (data['close'] - data['open']) / data['open']
    data['hl_close_pattern'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-day Momentum Quality
    data['momentum_quality'] = data['ret_3d'].rolling(5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else 0
    )
    data['momentum_decay'] = data['ret_1d'] / (data['ret_3d'].abs().rolling(3).mean() + 1e-8)
    
    # Combine factors with weights
    factors = [
        'range_momentum', 'gap_efficiency', 'gap_filling', 'momentum_persistence',
        'momentum_magnitude', 'high_amount_momentum', 'volume_confirmation',
        'volume_divergence', 'open_volume_ratio', 'volume_distribution',
        'high_vol_momentum', 'early_momentum', 'late_momentum', 'hl_close_pattern',
        'momentum_quality', 'momentum_decay'
    ]
    
    # Normalize each factor and combine
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for factor in factors:
        if factor in data.columns:
            # Z-score normalization within each day (cross-sectional)
            normalized = data.groupby(data.index)[factor].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            factor_values = factor_values.add(normalized.fillna(0), fill_value=0)
    
    return factor_values
