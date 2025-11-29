import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Price Efficiency Factor
    Combines opening auction efficiency, midday compression patterns, and closing auction dynamics
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic intraday metrics
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    
    # 1. Opening Auction Efficiency
    # Pre-market imbalance proxy using opening volume concentration
    data['opening_volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['avg_opening_ratio_10d'] = data['opening_volume_ratio'].rolling(window=10, min_periods=5).mean()
    data['opening_imbalance'] = data['opening_volume_ratio'] / data['avg_opening_ratio_10d']
    
    # Opening range efficiency (proxy for first 30 minutes)
    data['opening_range_efficiency'] = np.where(
        abs(data['overnight_gap']) > 0.001,
        data['daily_range'] / abs(data['overnight_gap']),
        1.0
    )
    
    # Opening efficiency score
    data['opening_efficiency'] = (
        (1 / (1 + abs(data['opening_imbalance'] - 1))) * 0.6 +
        (1 / (1 + data['opening_range_efficiency'])) * 0.4
    )
    
    # 2. Midday Price Compression Patterns
    # Midday range compression (proxy using intraday volatility)
    data['midday_volatility'] = data['high'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    data['daily_volatility'] = data['close'].pct_change().rolling(window=10).std()
    data['compression_ratio'] = data['midday_volatility'] / (data['daily_volatility'] + 1e-8)
    
    # VWAP deviation during compression periods
    data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
    data['vwap_deviation'] = abs(data['close'] - data['vwap']) / data['close']
    data['avg_vwap_deviation'] = data['vwap_deviation'].rolling(window=5).mean()
    
    # Midday efficiency score
    data['midday_efficiency'] = (
        (1 / (1 + data['compression_ratio'])) * 0.5 +
        (1 / (1 + data['vwap_deviation'] / (data['avg_vwap_deviation'] + 1e-8))) * 0.5
    )
    
    # 3. Closing Auction Dynamics
    # Closing volume concentration (proxy using final hour)
    data['closing_volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['avg_closing_ratio_10d'] = data['closing_volume_ratio'].rolling(window=10, min_periods=5).mean()
    data['closing_imbalance'] = data['closing_volume_ratio'] / data['avg_closing_ratio_10d']
    
    # Closing price impact
    data['closing_vwap_deviation'] = abs(data['close'] - data['vwap']) / data['close']
    data['avg_closing_vwap_dev'] = data['closing_vwap_deviation'].rolling(window=5).mean()
    
    # Closing efficiency score
    data['closing_efficiency'] = (
        (1 / (1 + abs(data['closing_imbalance'] - 1))) * 0.6 +
        (1 / (1 + data['closing_vwap_deviation'] / (data['avg_closing_vwap_dev'] + 1e-8))) * 0.4
    )
    
    # 4. Combine Components with Dynamic Weights
    # Market regime detection using volatility
    data['volatility_regime'] = data['daily_volatility'].rolling(window=10).mean()
    data['volatility_quantile'] = data['volatility_regime'].rolling(window=20).apply(
        lambda x: pd.qcut(x, 3, labels=False, duplicates='drop').iloc[-1] if len(x) == 20 else 1, 
        raw=False
    )
    
    # Dynamic weighting based on volatility regime
    def calculate_dynamic_weights(row):
        if pd.isna(row['volatility_quantile']):
            return [0.4, 0.35, 0.25]  # Default weights
        
        regime = row['volatility_quantile']
        if regime == 0:  # Low volatility
            return [0.3, 0.5, 0.2]    # Emphasize midday compression
        elif regime == 2:  # High volatility
            return [0.45, 0.25, 0.3]  # Emphasize opening and closing
        else:  # Normal volatility
            return [0.4, 0.35, 0.25]  # Balanced weights
    
    # Apply dynamic weights
    weights = data.apply(calculate_dynamic_weights, axis=1, result_type='expand')
    weights.columns = ['w_open', 'w_mid', 'w_close']
    
    # Final efficiency score
    data['intraday_efficiency_score'] = (
        data['opening_efficiency'] * weights['w_open'] +
        data['midday_efficiency'] * weights['w_mid'] +
        data['closing_efficiency'] * weights['w_close']
    )
    
    # Normalize the final score
    data['efficiency_factor'] = (
        data['intraday_efficiency_score'] - 
        data['intraday_efficiency_score'].rolling(window=20).mean()
    ) / (data['intraday_efficiency_score'].rolling(window=20).std() + 1e-8)
    
    return data['efficiency_factor']
