import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate basic price and volume metrics
    data['range'] = data['high'] - data['low']
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['price_change'] = data['close'] - data['open']
    data['dollar_volume'] = data['close'] * data['volume']
    
    # Short-term Range Momentum (3-day)
    # Range Expansion/Contraction Pattern
    data['range_change'] = data['range'] / data['range'].shift(1)
    data['range_expansion'] = (data['range_change'] > 1.1).astype(int)
    data['range_contraction'] = (data['range_change'] < 0.9).astype(int)
    
    # Volume-Weighted Momentum Quality
    data['volume_change'] = data['volume'] / data['volume'].shift(1)
    data['volume_surge'] = (data['volume_change'] > 1.2).astype(int)
    data['volume_drying'] = (data['volume_change'] < 0.8).astype(int)
    
    # Range persistence with price confirmation
    data['price_confirmation'] = np.where(
        (data['close'] > data['open']) & (data['range_expansion'] == 1), 1,
        np.where((data['close'] < data['open']) & (data['range_contraction'] == 1), -1, 0)
    )
    
    # Short-term momentum score (3-day window)
    data['short_term_momentum'] = (
        data['range_expansion'].rolling(window=3, min_periods=1).mean() * 
        data['price_confirmation'].rolling(window=3, min_periods=1).mean() *
        (1 + data['volume_surge'].rolling(window=3, min_periods=1).mean())
    )
    
    # Medium-term Range Momentum (10-day)
    # Range Efficiency Persistence
    data['range_efficiency'] = data['range'] / data['mid_price']
    data['range_efficiency_trend'] = data['range_efficiency'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Volume trend alignment
    data['volume_trend'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Gap Momentum Integration
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_strength'] = abs(data['opening_gap']) / data['range_efficiency']
    data['gap_filling_efficiency'] = np.where(
        data['opening_gap'] > 0,
        (data['high'] - data['open']) / data['opening_gap'],
        (data['open'] - data['low']) / abs(data['opening_gap'])
    )
    
    # Medium-term momentum score
    data['medium_term_momentum'] = (
        data['range_efficiency_trend'] * 
        np.sign(data['volume_trend']) *
        data['gap_filling_efficiency'].rolling(window=10, min_periods=5).mean()
    )
    
    # Amount-Flow Momentum Synchronization
    # Dollar Volume Impact Efficiency
    data['price_change_per_amount'] = data['price_change'] / (data['amount'] + 1e-8)
    data['range_per_amount'] = data['range'] / (data['amount'] + 1e-8)
    
    # Amount Flow Timing Patterns
    data['opening_amount_ratio'] = data['amount'].rolling(window=5, min_periods=3).apply(
        lambda x: x[0] / np.mean(x[1:]) if len(x) > 1 else 1, raw=True
    )
    
    # Amount-Momentum Regime Detection
    data['amount_quantile'] = data['amount'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
    )
    
    data['efficient_flow'] = np.where(
        (data['amount_quantile'] > 0.7) & (data['short_term_momentum'] > 0),
        data['price_change_per_amount'] * data['short_term_momentum'],
        0
    )
    
    data['inefficient_flow'] = np.where(
        (data['amount_quantile'] < 0.3) & (data['short_term_momentum'] < 0),
        data['price_change_per_amount'] * data['short_term_momentum'],
        0
    )
    
    # Amount-Flow Efficiency Score
    data['amount_efficiency'] = (
        data['efficient_flow'].rolling(window=5, min_periods=3).mean() -
        data['inefficient_flow'].rolling(window=5, min_periods=3).mean()
    )
    
    # Multi-Dimensional Momentum Quality
    # Range-Volume Synchronization Score
    data['range_volume_sync'] = (
        data['short_term_momentum'].rolling(window=5, min_periods=3).corr(
            data['volume_trend'].rolling(window=5, min_periods=3).mean()
        ) *
        np.sign(data['medium_term_momentum'])
    )
    
    # Final Cross-Sectional Efficiency Factor
    data['efficiency_factor'] = (
        data['short_term_momentum'].fillna(0) * 0.3 +
        data['medium_term_momentum'].fillna(0) * 0.4 +
        data['amount_efficiency'].fillna(0) * 0.2 +
        data['range_volume_sync'].fillna(0) * 0.1
    )
    
    # Apply cross-sectional ranking (z-score normalization within each day)
    def cross_sectional_rank(series):
        if len(series) > 1:
            return (series - series.mean()) / (series.std() + 1e-8)
        else:
            return series * 0
    
    # Group by date and apply cross-sectional normalization
    result = data.groupby(data.index)['efficiency_factor'].transform(cross_sectional_rank)
    
    return result
