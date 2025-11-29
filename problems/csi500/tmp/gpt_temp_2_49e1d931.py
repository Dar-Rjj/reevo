import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining efficiency, volume, gap, and institutional flow signals
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Pre-calculate common metrics
    data['prev_close'] = data['close'].shift(1)
    data['daily_range_pct'] = (data['high'] - data['low']) / data['prev_close']
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Timeframe Efficiency Momentum
    # 3-day and 10-day efficiency averages
    data['eff_3d'] = data['intraday_efficiency'].rolling(window=3, min_periods=2).mean()
    data['eff_10d'] = data['intraday_efficiency'].rolling(window=10, min_periods=5).mean()
    
    # Efficiency momentum components
    data['eff_divergence'] = data['eff_3d'] - data['eff_10d']
    data['eff_momentum'] = data['eff_3d'] - data['eff_3d'].shift(1)
    
    # Volatility-adjusted efficiency quality
    data['range_5d_avg'] = data['daily_range_pct'].rolling(window=5, min_periods=3).mean()
    data['eff_quality'] = data['intraday_efficiency'].abs() / data['daily_range_pct'].replace(0, np.nan)
    
    # Volume-Range Convergence Framework
    data['range_momentum'] = data['daily_range_pct'] / data['range_5d_avg'].replace(0, np.nan)
    
    # Volume momentum hierarchy
    data['volume_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['vol_momentum'] = (data['volume'] - data['volume_5d']) / data['volume_5d'].replace(0, np.nan)
    data['vol_acceleration'] = (data['volume_5d'] - data['volume_10d']) / data['volume_10d'].replace(0, np.nan)
    
    # Range-volume convergence signals
    data['range_vol_alignment'] = np.where(
        (data['range_momentum'] > 1) & (data['vol_momentum'] > 0),
        data['range_momentum'] * data['vol_momentum'],
        0
    )
    data['inefficient_action'] = np.where(
        (data['daily_range_pct'] > data['range_5d_avg']) & (data['vol_momentum'] < 0),
        -data['daily_range_pct'],
        0
    )
    data['compression_buildup'] = np.where(
        (data['daily_range_pct'] < data['range_5d_avg']) & (data['vol_momentum'] > 0),
        data['vol_momentum'],
        0
    )
    
    # Gap Persistence Momentum System
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close'].replace(0, np.nan)
    
    # Intraday gap behavior
    data['gap_fill_ratio'] = np.where(
        data['overnight_gap'] > 0,
        (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan),
        (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    )
    data['gap_persistence'] = 1 - data['gap_fill_ratio']
    
    # Gap-volume confirmation
    data['gap_vol_confirmation'] = data['gap_persistence'] * np.sign(data['overnight_gap']) * data['vol_momentum']
    
    # Institutional Flow Pressure Integration
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['pressure_3d'] = data['buying_pressure'].rolling(window=3, min_periods=2).mean()
    data['pressure_momentum'] = data['pressure_3d'] - data['pressure_3d'].shift(1)
    
    # Amount-weighted pressure
    data['amount_per_volume'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['weighted_pressure'] = data['buying_pressure'] * data['amount_per_volume']
    data['weighted_pressure_3d'] = data['weighted_pressure'].rolling(window=3, min_periods=2).mean()
    
    # Pressure-gap alignment
    data['pressure_gap_alignment'] = np.where(
        np.sign(data['overnight_gap']) == np.sign(data['weighted_pressure_3d']),
        data['weighted_pressure_3d'] * data['gap_persistence'],
        -data['weighted_pressure_3d'] * data['gap_persistence']
    )
    
    # Cross-Sectional Composite Generation
    # Combine all components
    components = [
        data['eff_divergence'],           # Efficiency momentum
        data['eff_momentum'],             # Efficiency change rate
        data['eff_quality'],              # Volatility-adjusted efficiency
        data['range_vol_alignment'],      # Valid breakout signals
        data['compression_buildup'],      # Compression signals
        data['gap_vol_confirmation'],     # Gap persistence with volume
        data['pressure_gap_alignment'],   # Institutional flow alignment
        data['weighted_pressure_3d']      # Institutional pressure
    ]
    
    # Standardize each component (z-score)
    factor_components = []
    for component in components:
        # Remove outliers and calculate z-score
        component_clean = component.replace([np.inf, -np.inf], np.nan)
        mean_val = component_clean.rolling(window=20, min_periods=10).mean()
        std_val = component_clean.rolling(window=20, min_periods=10).std()
        z_score = (component_clean - mean_val) / std_val.replace(0, np.nan)
        factor_components.append(z_score.fillna(0))
    
    # Equal weighted combination
    composite_factor = sum(factor_components) / len(factor_components)
    
    # Final cross-sectional ranking (z-score across universe for each day)
    def cross_sectional_rank(series):
        return (series - series.mean()) / series.std() if series.std() != 0 else 0
    
    final_factor = composite_factor.groupby(composite_factor.index).transform(cross_sectional_rank)
    
    return final_factor
