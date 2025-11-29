import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor based on price-volume asymmetry and regime persistence
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic daily metrics
    data['daily_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['prev_close'] = data['close'].shift(1)
    
    # 1. Price Distribution Asymmetry
    # Opening price position
    data['open_position'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Closing price momentum
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['close_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # 2. Volume Concentration & Timing
    # We'll use amount as proxy for dollar volume since we don't have intraday volume breakdown
    data['daily_volume'] = data['volume']
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Volume-price efficiency
    data['volume_price_eff'] = (data['close'] - data['open']).abs() / (data['volume'] + 1e-10)
    
    # 3. Regime Classification
    # Volatility regime
    data['volatility_regime'] = data['daily_range'].rolling(window=10, min_periods=5).mean()
    
    # Trend regime - 5-day momentum
    data['trend_strength'] = data['close'].pct_change(5)
    data['trend_persistence'] = ((data['close'] > data['close'].shift(1)).rolling(window=5).sum() - 
                                (data['close'] < data['close'].shift(1)).rolling(window=5).sum()) / 5
    
    # 4. Cross-sectional calculations (within day)
    def cross_sectional_rank(group):
        return group.rank(pct=True)
    
    # Daily cross-sectional rankings
    daily_features = ['open_position', 'close_position', 'volume_concentration', 
                     'volume_price_eff', 'trend_persistence']
    
    for feature in daily_features:
        data[f'{feature}_cs_rank'] = data.groupby(data.index)[feature].transform(cross_sectional_rank)
    
    # 5. Persistence metrics
    # Signal persistence over 3 days
    persistence_windows = [3, 5]
    for window in persistence_windows:
        for feature in ['open_position_cs_rank', 'close_position_cs_rank']:
            data[f'{feature}_persist_{window}'] = (
                data[feature].rolling(window=window).std().fillna(0.5)  # Lower std = more persistent
            )
    
    # 6. Integrated alpha construction
    # Core asymmetry signals
    data['price_asymmetry'] = (
        0.4 * data['open_position_cs_rank'] + 
        0.4 * data['close_position_cs_rank'] + 
        0.2 * (1 - data['open_position_cs_rank_persist_3'])
    )
    
    data['volume_asymmetry'] = (
        0.6 * data['volume_concentration_cs_rank'] + 
        0.4 * (1 - data['volume_price_eff_cs_rank'])  # Inverse relationship
    )
    
    # Regime adjustment
    volatility_weight = 1 / (1 + data['volatility_regime'].rolling(window=10).rank(pct=True))
    trend_weight = data['trend_persistence_cs_rank'].abs()
    
    # Final alpha factor
    data['alpha_raw'] = (
        volatility_weight * data['price_asymmetry'] + 
        (1 - volatility_weight) * data['volume_asymmetry'] +
        0.1 * trend_weight
    )
    
    # Normalize and handle outliers
    alpha = data['alpha_raw'].copy()
    alpha = alpha.groupby(alpha.index).transform(lambda x: (x - x.mean()) / x.std())
    alpha = np.clip(alpha, -3, 3)  # Winsorize at 3 std
    
    return alpha
