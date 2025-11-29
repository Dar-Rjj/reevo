import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining price-volume divergence, range expansion,
    opening auction dynamics, amount flow concentration, and volatility regime transitions.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Patterns
    # Intraday momentum vs volume flow
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['intraday_volume_flow'] = data['volume'] * np.sign(data['close'] - data['open'])
    data['efficiency_volume_concentration'] = data['intraday_efficiency'] * data['intraday_volume_flow']
    
    # Multi-day divergence detection
    data['ret_5d'] = data['close'].pct_change(5)
    data['vol_change_5d'] = data['volume'].pct_change(5)
    data['ret_20d'] = data['close'].pct_change(20)
    data['vol_change_20d'] = data['volume'].pct_change(20)
    
    # Rolling correlations for return-volume divergence
    data['corr_5d'] = data['close'].pct_change().rolling(5).corr(data['volume'].pct_change())
    data['corr_20d'] = data['close'].pct_change().rolling(20).corr(data['volume'].pct_change())
    
    # Range Expansion Momentum
    # Dynamic range analysis
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_5d_avg'] = data['daily_range'].rolling(5).mean()
    data['range_expansion'] = data['daily_range'] / (data['range_5d_avg'] + 1e-8)
    
    # Breakout confirmation
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['volume_range_expansion'] = data['volume'] * data['range_expansion']
    
    # Opening Auction Dynamics (using open price as proxy)
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = data['opening_gap'] * np.sign(data['close'] - data['open'])
    data['opening_range'] = (data['high'] - data['low']) / data['open']
    
    # Opening price relative to previous day's range
    data['prev_day_range'] = (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)
    data['opening_position'] = (data['open'] - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    
    # Amount Flow Concentration
    data['amount_volatility'] = data['amount'].pct_change().rolling(5).std()
    data['price_volatility'] = data['close'].pct_change().rolling(5).std()
    data['amount_vol_ratio'] = data['amount_volatility'] / (data['price_volatility'] + 1e-8)
    
    # Large amount days detection
    data['amount_5d_avg'] = data['amount'].rolling(5).mean()
    data['large_amount_day'] = data['amount'] / (data['amount_5d_avg'] + 1e-8)
    
    # Consecutive high-amount days
    data['high_amount_flag'] = (data['amount'] > data['amount_5d_avg']).astype(int)
    data['consecutive_high_amount'] = data['high_amount_flag'].rolling(3).sum()
    
    # Volatility Regime Transitions
    data['volatility_5d'] = data['close'].pct_change().rolling(5).std()
    data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
    data['volatility_regime'] = data['volatility_5d'] / (data['volatility_20d'] + 1e-8)
    
    # Volume behavior during volatility shifts
    data['volume_volatility'] = data['volume'].pct_change().rolling(5).std()
    data['volume_vol_regime'] = data['volume_volatility'] * data['volatility_regime']
    
    # Price range behavior across volatility states
    data['range_volatility_ratio'] = data['daily_range'] / (data['volatility_5d'] + 1e-8)
    
    # Combine factors with appropriate weights
    factors = [
        # Price-Volume Divergence (30%)
        -0.3 * data['corr_5d'].fillna(0),  # Negative correlation suggests divergence
        -0.3 * data['corr_20d'].fillna(0),
        0.2 * data['efficiency_volume_concentration'].fillna(0),
        
        # Range Expansion Momentum (25%)
        0.25 * data['range_expansion'].fillna(0),
        0.15 * data['close_position'].fillna(0),
        0.1 * data['volume_range_expansion'].fillna(0),
        
        # Opening Auction Dynamics (20%)
        0.2 * data['gap_persistence'].fillna(0),
        0.15 * data['opening_position'].fillna(0),
        
        # Amount Flow Concentration (15%)
        0.15 * data['amount_vol_ratio'].fillna(0),
        0.1 * data['consecutive_high_amount'].fillna(0),
        
        # Volatility Regime Transitions (10%)
        0.1 * data['volatility_regime'].fillna(0),
        0.05 * data['range_volatility_ratio'].fillna(0)
    ]
    
    # Calculate final factor
    factor_series = sum(factors)
    
    # Normalize by cross-sectional z-score
    def cross_sectional_zscore(series):
        return (series - series.mean()) / (series.std() + 1e-8)
    
    # Apply cross-sectional normalization
    factor_normalized = factor_series.groupby(factor_series.index).transform(cross_sectional_zscore)
    
    return factor_normalized
