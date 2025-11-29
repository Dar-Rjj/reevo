import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Price-Volume Efficiency Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Efficiency Metrics
    data['price_efficiency_ratio'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_concentration_efficiency'] = data['volume'] / data['amount']
    data['daily_capture_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Replace infinities and handle division by zero
    efficiency_metrics = ['price_efficiency_ratio', 'volume_concentration_efficiency', 'daily_capture_efficiency']
    for metric in efficiency_metrics:
        data[metric] = data[metric].replace([np.inf, -np.inf], np.nan)
        data[metric] = data[metric].fillna(0)
    
    # Multi-timeframe Efficiency Divergence
    data['short_term_efficiency'] = data['price_efficiency_ratio'].rolling(window=3, min_periods=1).mean()
    data['medium_term_efficiency'] = data['daily_capture_efficiency'].rolling(window=10, min_periods=1).mean()
    data['efficiency_divergence'] = data['short_term_efficiency'] - data['medium_term_efficiency']
    
    # Volume-Price Efficiency Relationship
    data['volume_quantile'] = data.groupby(data.index)['volume'].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')
    )
    data['efficiency_quantile'] = data.groupby(data.index)['price_efficiency_ratio'].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')
    )
    
    # High volume efficiency persistence
    high_volume_mask = data['volume_quantile'] == 3
    data['high_volume_efficiency_persistence'] = data['price_efficiency_ratio'].rolling(window=5, min_periods=1).mean()
    data.loc[~high_volume_mask, 'high_volume_efficiency_persistence'] = 0
    
    # Low volume efficiency reversal
    low_volume_mask = data['volume_quantile'] == 0
    data['low_volume_efficiency_reversal'] = -data['price_efficiency_ratio'].rolling(window=3, min_periods=1).mean()
    data.loc[~low_volume_mask, 'low_volume_efficiency_reversal'] = 0
    
    # Efficiency Regime Detection
    data['efficiency_volatility'] = data['price_efficiency_ratio'].rolling(window=20, min_periods=1).std()
    data['efficiency_regime'] = (data['price_efficiency_ratio'] > data['price_efficiency_ratio'].rolling(window=20, min_periods=1).mean()).astype(int)
    data['regime_transition'] = data['efficiency_regime'].diff()
    
    # Efficiency-Momentum Interaction
    data['price_momentum'] = data['close'].pct_change(periods=5)
    data['efficiency_adjusted_momentum'] = data['price_momentum'] * data['price_efficiency_ratio']
    
    # Dynamic Efficiency Scoring - Multi-dimensional composite
    efficiency_factors = [
        'price_efficiency_ratio',
        'daily_capture_efficiency', 
        'efficiency_divergence',
        'high_volume_efficiency_persistence',
        'low_volume_efficiency_reversal',
        'efficiency_adjusted_momentum'
    ]
    
    # Z-score normalization for each factor cross-sectionally
    for factor in efficiency_factors:
        zscore_col = f'{factor}_zscore'
        data[zscore_col] = data.groupby(data.index)[factor].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        data[zscore_col] = data[zscore_col].fillna(0)
    
    # Composite efficiency score with equal weights
    zscore_cols = [f'{factor}_zscore' for factor in efficiency_factors]
    data['composite_efficiency_score'] = data[zscore_cols].mean(axis=1)
    
    # Cross-sectional percentile ranking
    data['efficiency_alpha'] = data.groupby(data.index)['composite_efficiency_score'].transform(
        lambda x: pd.qcut(x, 100, labels=False, duplicates='drop') / 100.0
    )
    
    # Final alpha factor
    alpha_series = data['efficiency_alpha']
    
    return alpha_series
