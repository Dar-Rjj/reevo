import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Path Efficiency Momentum factor
    Combines intraday volume distribution analysis with price path efficiency scoring
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday returns and metrics
    data['intra_high_low'] = data['high'] - data['low']
    data['abs_intra_return'] = abs(data['close'] - data['open'])
    
    # Estimate session boundaries (assuming 6.5 hour trading day)
    # First hour: 9:30-10:30, Last hour: 15:00-16:00
    # Morning: 9:30-12:00, Afternoon: 12:00-16:00
    # Midday: 12:00
    
    # Session volume ratios (using amount as proxy for dollar volume)
    data['first_hour_volume_ratio'] = data['amount'].rolling(window=5).apply(
        lambda x: x.iloc[0] / x.sum() if x.sum() > 0 else 0, raw=False
    )
    data['last_hour_volume_ratio'] = data['amount'].rolling(window=5).apply(
        lambda x: x.iloc[-1] / x.sum() if x.sum() > 0 else 0, raw=False
    )
    
    # Volume profile skewness (morning concentration)
    data['morning_volume_concentration'] = data['amount'].rolling(window=10).apply(
        lambda x: x[:5].sum() / x.sum() if x.sum() > 0 else 0, raw=False
    )
    
    # Price path efficiency calculations
    data['actual_path_length'] = (
        abs(data['high'] - data['open']) + 
        abs(data['low'] - data['open']) + 
        abs(data['close'] - data['open'])
    )
    data['net_movement'] = abs(data['close'] - data['open'])
    data['path_efficiency_ratio'] = np.where(
        data['actual_path_length'] > 0,
        data['net_movement'] / data['actual_path_length'],
        0
    )
    
    # Session-specific efficiency (using rolling windows to estimate session patterns)
    data['morning_efficiency'] = data['path_efficiency_ratio'].rolling(window=5).apply(
        lambda x: np.mean(x[:3]) if len(x) >= 3 else np.nan, raw=False
    )
    data['afternoon_efficiency'] = data['path_efficiency_ratio'].rolling(window=5).apply(
        lambda x: np.mean(x[3:]) if len(x) >= 5 else np.nan, raw=False
    )
    
    # Volume-weighted path efficiency
    data['volume_weighted_efficiency'] = (
        data['path_efficiency_ratio'] * data['amount'] / 
        data['amount'].rolling(window=10).mean()
    )
    
    # Volume timing alignment
    data['volume_path_alignment'] = (
        data['first_hour_volume_ratio'] * data['morning_efficiency'] +
        data['last_hour_volume_ratio'] * data['afternoon_efficiency']
    )
    
    # Multi-session momentum integration
    data['volume_distribution_persistence'] = (
        data['morning_volume_concentration'].rolling(window=3).std() * -1  # Lower std = more persistent
    )
    
    data['efficiency_consistency'] = (
        data['path_efficiency_ratio'].rolling(window=5).std() * -1  # Lower std = more consistent
    )
    
    # Cross-session momentum building
    data['volume_accumulation_trend'] = data['amount'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0, raw=False
    )
    
    data['efficiency_momentum'] = data['path_efficiency_ratio'].diff(periods=1).rolling(window=3).mean()
    
    # Final factor combination
    factor = (
        0.25 * data['volume_path_alignment'].fillna(0) +
        0.20 * data['volume_weighted_efficiency'].fillna(0) +
        0.15 * data['volume_distribution_persistence'].fillna(0) +
        0.15 * data['efficiency_consistency'].fillna(0) +
        0.15 * data['volume_accumulation_trend'].fillna(0) +
        0.10 * data['efficiency_momentum'].fillna(0)
    )
    
    # Remove any potential lookahead bias and ensure proper indexing
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return factor
