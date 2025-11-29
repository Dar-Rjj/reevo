import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum-Liquidity Integration Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Cross-Timeframe Momentum Efficiency
    # Intraday efficiency
    data['intraday_eff'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Overnight efficiency
    data['overnight_eff'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-day efficiency persistence
    data['eff_3d_persistence'] = data['intraday_eff'].rolling(window=3).std()
    data['eff_5d_persistence'] = data['intraday_eff'].rolling(window=5).std()
    
    # Efficiency divergence
    data['eff_divergence'] = data['intraday_eff'] - data['overnight_eff']
    
    # 2. Volume Absorption Dynamics
    # Key level absorption intensity
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['mid_volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Absorption persistence
    data['absorption_persistence'] = data['mid_volume_ratio'].rolling(window=5).mean()
    
    # Volume concentration during efficiency extremes
    high_eff_mask = data['intraday_eff'] > data['intraday_eff'].rolling(window=20).quantile(0.8)
    low_eff_mask = data['intraday_eff'] < data['intraday_eff'].rolling(window=20).quantile(0.2)
    
    data['volume_concentration_high_eff'] = data['volume'].rolling(window=5).std() * high_eff_mask
    data['volume_concentration_low_eff'] = data['volume'].rolling(window=5).std() * low_eff_mask
    
    # 3. Price Elasticity with Volume Validation
    # Momentum response elasticity
    data['eff_change'] = data['intraday_eff'].diff()
    data['volume_shock'] = data['volume'] / data['volume'].rolling(window=20).mean() - 1
    data['elasticity'] = np.abs(data['eff_change']) / (np.abs(data['volume_shock']) + 1e-6)
    
    # Cross-timeframe elasticity consistency
    data['elasticity_consistency'] = data['elasticity'].rolling(window=5).std()
    
    # 4. Microstructural Efficiency Gradient
    # Price path efficiency
    data['price_path'] = np.abs(data['close'] - data['open']) + np.abs(data['high'] - data['low'])
    data['micro_eff'] = data['price_path'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Efficiency-absorption validation
    data['eff_abs_validation'] = data['micro_eff'] * data['mid_volume_ratio']
    
    # 5. Temporal Momentum-Liquidity Integration
    # Momentum persistence with volume confirmation
    data['momentum_persistence'] = data['intraday_eff'].rolling(window=5).corr(data['volume'].rolling(window=5).mean())
    
    # Volume distribution during momentum extremes
    data['volume_dist_momentum'] = data['volume'].rolling(window=5).std() * np.abs(data['intraday_eff'])
    
    # 6. Integrated Factor Construction
    # Primary efficiency score
    eff_weights = [0.4, 0.3, 0.3]  # intraday, overnight, micro
    data['primary_eff_score'] = (
        eff_weights[0] * data['intraday_eff'].fillna(0) +
        eff_weights[1] * data['overnight_eff'].fillna(0) +
        eff_weights[2] * data['micro_eff'].fillna(0)
    )
    
    # Volume absorption multiplier
    data['absorption_multiplier'] = (
        data['mid_volume_ratio'].fillna(1) *
        data['absorption_persistence'].fillna(1) *
        (1 + data['volume_concentration_high_eff'].fillna(0) - data['volume_concentration_low_eff'].fillna(0))
    )
    
    # Elasticity confirmation filter
    data['elasticity_filter'] = 1 / (1 + np.exp(-data['elasticity'].fillna(0) / data['elasticity'].rolling(window=20).std().replace(0, 1)))
    
    # Microstructural efficiency adjustment
    data['micro_adjustment'] = np.tanh(data['eff_abs_validation'].fillna(0))
    
    # Final composite factor
    data['composite_factor'] = (
        data['primary_eff_score'] *
        data['absorption_multiplier'] *
        data['elasticity_filter'] *
        data['micro_adjustment']
    )
    
    # 7. Cross-Sectional Normalization
    # Z-score normalization within each day
    def cross_sectional_zscore(series):
        return (series - series.mean()) / (series.std() + 1e-6)
    
    # Apply cross-sectional normalization
    factor_series = data.groupby(data.index)['composite_factor'].transform(cross_sectional_zscore)
    
    return factor_series
