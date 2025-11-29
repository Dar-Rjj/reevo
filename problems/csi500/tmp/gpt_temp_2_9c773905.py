import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price Efficiency with Volume Confirmation factor
    Analyzes price efficiency across short and medium-term horizons with volume-based confirmation
    """
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['open_return'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    data['close_return'] = data['close'].pct_change()
    
    # 1. Short-term Price Efficiency
    # Opening Gap Efficiency
    data['gap_efficiency'] = np.where(
        data['daily_range'] > 0,
        np.abs(data['open_return']) / data['daily_range'],
        0
    )
    
    # Intraday Price Discovery Efficiency
    data['intraday_move'] = np.abs(data['close'] - data['open']) / data['open']
    data['volume_efficiency'] = np.where(
        data['volume'] > 0,
        data['intraday_move'] / (data['volume'] / data['volume'].rolling(20).mean()),
        0
    )
    
    # 2. Medium-term Efficiency Patterns
    # Multi-day price path efficiency
    data['range_5d'] = data['daily_range'].rolling(5).mean()
    data['return_5d'] = data['close_return'].rolling(5).sum()
    data['path_efficiency'] = np.where(
        data['range_5d'] > 0,
        np.abs(data['return_5d']) / data['range_5d'],
        0
    )
    
    # Efficiency regime changes
    data['efficiency_std'] = data['gap_efficiency'].rolling(10).std()
    data['efficiency_trend'] = data['gap_efficiency'].rolling(5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    # 3. Compare Efficiency Across Timeframes
    data['efficiency_divergence'] = (
        data['gap_efficiency'].rolling(5).mean() - 
        data['path_efficiency'].rolling(5).mean()
    )
    
    # Efficiency convergence
    data['efficiency_correlation'] = data['gap_efficiency'].rolling(10).corr(
        data['path_efficiency'].rolling(10).mean()
    )
    
    # 4. Volume-Based Confirmation
    # Volume surprise metrics
    data['volume_ma_20'] = data['volume'].rolling(20).mean()
    data['volume_surprise'] = (data['volume'] - data['volume_ma_20']) / data['volume_ma_20']
    data['volume_persistence'] = data['volume_surprise'].rolling(5).mean()
    
    # Volume-Price relationship
    data['vw_return'] = data['close_return'] * (data['volume'] / data['volume_ma_20'])
    data['volume_price_divergence'] = np.where(
        (np.abs(data['close_return']) > data['close_return'].rolling(20).std()) & 
        (data['volume'] < data['volume_ma_20']),
        -1,  # Fragile moves
        np.where(
            (np.abs(data['close_return']) < data['close_return'].rolling(20).std()) & 
            (data['volume'] > data['volume_ma_20'] * 1.5),
            1,  # Accumulation/Distribution
            0
        )
    )
    
    # 5. Combine Efficiency Analysis with Volume Confirmation
    # Primary efficiency signal
    data['primary_efficiency'] = (
        0.4 * data['gap_efficiency'].rolling(5).mean() +
        0.3 * data['path_efficiency'].rolling(5).mean() +
        0.3 * data['efficiency_correlation']
    )
    
    # Volume confidence adjustment
    data['volume_confidence'] = (
        0.5 * data['volume_persistence'] +
        0.3 * np.sign(data['volume_price_divergence']) +
        0.2 * np.tanh(data['volume_surprise'])
    )
    
    # Composite alpha factor
    data['alpha_factor'] = (
        data['primary_efficiency'] * 
        (1 + data['volume_confidence'])
    )
    
    # Final factor with normalization
    factor = data['alpha_factor'].copy()
    factor = (factor - factor.rolling(63).mean()) / (factor.rolling(63).std() + 1e-8)
    
    return factor
