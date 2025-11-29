import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Regime-Adaptive Price-Volume Asymmetry Factor
    """
    data = df.copy()
    
    # Calculate daily returns and price movements
    data['returns'] = data['close'].pct_change()
    data['price_change'] = data['close'] - data['open']
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    
    # 1. Multi-Horizon Volume Asymmetry
    # 1-day vs 5-day volume concentration
    data['up_volume_1d'] = np.where(data['returns'] > 0, data['volume'], 0)
    data['down_volume_1d'] = np.where(data['returns'] < 0, data['volume'], 0)
    
    # Rolling volume asymmetry ratios
    data['up_volume_5d'] = data['up_volume_1d'].rolling(window=5, min_periods=3).sum()
    data['down_volume_5d'] = data['down_volume_1d'].rolling(window=5, min_periods=3).sum()
    
    # Volume concentration divergence
    data['vol_conc_1d'] = (data['up_volume_1d'] - data['down_volume_1d']) / (data['up_volume_1d'] + data['down_volume_1d'] + 1e-8)
    data['vol_conc_5d'] = (data['up_volume_5d'] - data['down_volume_5d']) / (data['up_volume_5d'] + data['down_volume_5d'] + 1e-8)
    
    # Volume asymmetry divergence
    data['vol_asym_divergence'] = data['vol_conc_1d'] - data['vol_conc_5d']
    
    # 2. Regime-dependent characteristics
    # Market regime using rolling volatility
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    data['vol_regime'] = np.where(data['volatility_20d'] > data['volatility_20d'].rolling(window=60, min_periods=30).median(), 1, 0)
    
    # Bull/bear regime using rolling returns
    data['market_trend_20d'] = data['close'].pct_change(20)
    data['trend_regime'] = np.where(data['market_trend_20d'] > 0, 1, -1)
    
    # Regime-adaptive volume asymmetry
    data['regime_vol_asym'] = data['vol_asym_divergence'] * data['trend_regime']
    data['high_vol_asym'] = data['vol_asym_divergence'] * (1 - data['vol_regime'])
    data['low_vol_asym'] = data['vol_asym_divergence'] * data['vol_regime']
    
    # 3. Microstructure-induced asymmetry anomalies
    # Open/close volume concentration differentials
    data['intraday_vol_ratio'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-8)
    data['price_vol_correlation'] = data['returns'].rolling(window=5, min_periods=3).corr(data['intraday_vol_ratio'])
    
    # Asymmetry reversal following large spreads
    data['spread_estimate'] = (data['high'] - data['low']) / data['close']
    data['large_spread'] = np.where(data['spread_estimate'] > data['spread_estimate'].rolling(window=20, min_periods=10).quantile(0.7), 1, 0)
    data['post_spread_asym'] = data['vol_conc_1d'].shift(1) * data['large_spread']
    
    # 4. Intraday regime transition dynamics
    # Morning vs afternoon proxy using early vs late trading
    data['early_trading_range'] = (data['high'].rolling(window=3, min_periods=2).max() - data['low'].rolling(window=3, min_periods=2).min()) / data['close']
    data['late_trading_range'] = (data['high'] - data['low']) / data['close']
    data['intraday_regime_shift'] = data['late_trading_range'] - data['early_trading_range']
    
    # Asymmetry regime persistence
    data['asym_persistence'] = data['vol_conc_1d'].rolling(window=5, min_periods=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    
    # 5. Price gap absorption asymmetry
    # Overnight gap vs intraday movement
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_movement'] = (data['close'] - data['open']) / data['open']
    data['gap_absorption'] = np.where(data['overnight_gap'] * data['intraday_movement'] < 0, 
                                    abs(data['intraday_movement']) / (abs(data['overnight_gap']) + 1e-8), 0)
    
    # Gap filling speed and completeness
    data['gap_fill_efficiency'] = data['gap_absorption'].rolling(window=10, min_periods=5).mean()
    
    # 6. Multi-dimensional asymmetry composite
    # Cross-sectional normalization components
    components = [
        'vol_asym_divergence',
        'regime_vol_asym', 
        'high_vol_asym',
        'low_vol_asym',
        'price_vol_correlation',
        'post_spread_asym',
        'asym_persistence',
        'gap_fill_efficiency'
    ]
    
    # Remove any infinite values and handle NaNs
    for col in components:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    
    # Create composite factor with regime-adaptive weighting
    factor_components = []
    
    # Volume asymmetry component (40% weight)
    vol_asym_component = (
        0.4 * data['vol_asym_divergence'] + 
        0.3 * data['regime_vol_asym'] + 
        0.3 * data['price_vol_correlation']
    )
    factor_components.append(vol_asym_component)
    
    # Microstructure component (30% weight)
    micro_component = (
        0.5 * data['post_spread_asym'] + 
        0.5 * data['asym_persistence']
    )
    factor_components.append(0.3 * micro_component)
    
    # Gap absorption component (30% weight)
    gap_component = data['gap_fill_efficiency']
    factor_components.append(0.3 * gap_component)
    
    # Combine all components
    composite_factor = sum(factor_components)
    
    # Final cross-sectional normalization
    final_factor = (composite_factor - composite_factor.rolling(window=20, min_periods=10).mean()) / (composite_factor.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return final_factor
