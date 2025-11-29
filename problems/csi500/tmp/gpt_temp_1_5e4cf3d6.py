import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Asymmetry with Intraday Regime Switching factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Basic price calculations
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Directional Volume-Price Efficiency Analysis
    # Asymmetric Volume Response Components
    data['up_day'] = data['close'] > data['open']
    data['down_day'] = data['close'] < data['open']
    
    # Up-day volume intensity
    up_volume = data['volume'].where(data['up_day'], 0)
    data['up_volume_ma'] = up_volume.rolling(window=5, min_periods=3).mean()
    
    # Down-day volume compression
    down_volume = data['volume'].where(data['down_day'], 0)
    data['down_volume_ma'] = down_volume.rolling(window=5, min_periods=3).mean()
    
    # Volume asymmetry ratio
    data['volume_asymmetry'] = data['up_volume_ma'] / (data['down_volume_ma'] + 1e-8)
    
    # Directional volume persistence
    data['volume_direction'] = np.where(data['up_day'], 1, np.where(data['down_day'], -1, 0))
    data['volume_persistence'] = (
        data['volume_direction'].rolling(window=3).apply(
            lambda x: len(set(x)) == 1 and x.iloc[0] != 0 if len(x) == 3 else np.nan
        )
    )
    
    # Price Efficiency Under Different Volume Regimes
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['high_volume'] = data['volume'] > data['volume_ma_5']
    data['low_volume'] = data['volume'] < data['volume_ma_5']
    
    # High-volume price efficiency
    data['high_vol_efficiency'] = np.where(
        data['high_volume'],
        abs(data['close'] - data['open']) / (data['true_range'] + 1e-8),
        np.nan
    )
    data['high_vol_efficiency_ma'] = data['high_vol_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Low-volume price drift
    data['low_vol_drift'] = np.where(
        data['low_volume'],
        data['close'] - data['open'],
        np.nan
    )
    data['low_vol_drift_ma'] = data['low_vol_drift'].rolling(window=5, min_periods=3).mean()
    
    # Amount-Based Liquidity Signals
    data['amount_ma'] = data['amount'].rolling(window=10, min_periods=5).mean()
    data['large_trade_days'] = (data['amount'] > 3 * data['amount_ma']).astype(int)
    data['large_trade_concentration'] = data['large_trade_days'].rolling(window=5, min_periods=3).mean()
    
    # Amount-volume divergence
    data['amount_volume_ratio'] = data['amount'] / (data['volume'] + 1e-8)
    data['amount_volume_divergence'] = data['amount_volume_ratio'] - data['amount_volume_ratio'].rolling(window=5, min_periods=3).mean()
    
    # Intraday Price Discovery Dynamics
    # Opening Auction Asymmetry
    data['opening_gap'] = data['open'] - data['prev_close']
    data['up_gap'] = data['opening_gap'] > 0
    data['down_gap'] = data['opening_gap'] < 0
    
    # Opening gap absorption (for up gaps)
    data['gap_absorption'] = np.where(
        data['up_gap'],
        (data['high'] - data['open']) / (data['opening_gap'] + 1e-8),
        np.nan
    )
    
    # Gap resistance (for down gaps)
    data['gap_resistance'] = np.where(
        data['down_gap'],
        (data['open'] - data['low']) / (-data['opening_gap'] + 1e-8),
        np.nan
    )
    
    # Estimate first hour volume (assuming first hour is 25% of daily volume)
    data['opening_volume_ratio'] = 0.25  # Conservative estimate
    
    # Closing Auction Dynamics
    data['eod_price_pressure'] = data['close'] - data['open']
    
    # Multi-Scale Regime Switching Framework
    # Volume regime classification
    data['volume_regime'] = np.where(
        data['volume'] > data['volume_ma_5'] * 1.2, 'high',
        np.where(data['volume'] < data['volume_ma_5'] * 0.8, 'low', 'normal')
    )
    
    # Price trend regime
    data['price_trend'] = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
    )
    
    # Regime persistence
    regime_map = {'high': 2, 'normal': 1, 'low': 0}
    data['volume_regime_code'] = data['volume_regime'].map(regime_map)
    data['regime_persistence'] = (
        data['volume_regime_code'].rolling(window=3).apply(
            lambda x: len(set(x)) == 1 if len(x) == 3 else np.nan
        )
    )
    
    # Asymmetric Response Factor Construction
    # Core directional volume efficiency
    data['up_vol_efficiency'] = np.where(
        data['up_day'],
        (data['close'] - data['open']) / (data['true_range'] + 1e-8),
        0
    )
    data['down_vol_resistance'] = np.where(
        data['down_day'],
        (data['open'] - data['close']) / (data['true_range'] + 1e-8),
        0
    )
    
    # Rolling averages for stability
    data['up_vol_efficiency_ma'] = data['up_vol_efficiency'].rolling(window=5, min_periods=3).mean()
    data['down_vol_resistance_ma'] = data['down_vol_resistance'].rolling(window=5, min_periods=3).mean()
    
    # Volume asymmetry momentum
    data['volume_asymmetry_momentum'] = data['volume_asymmetry'] - data['volume_asymmetry'].shift(3)
    
    # Intraday pattern integration
    data['opening_alignment'] = np.where(
        data['up_gap'],
        data['gap_absorption'].fillna(0),
        np.where(data['down_gap'], -data['gap_resistance'].fillna(0), 0)
    )
    
    # Final Alpha Implementation
    # Raw Asymmetric Response Factor
    data['core_directional_efficiency'] = (
        data['up_vol_efficiency_ma'] - data['down_vol_resistance_ma']
    )
    
    data['intraday_regime_alignment'] = (
        data['opening_alignment'] * data['eod_price_pressure']
    )
    
    data['multi_scale_confirmation'] = (
        data['volume_asymmetry_momentum'] * data['regime_persistence'].fillna(0)
    )
    
    # Combine components with appropriate weighting
    data['raw_asymmetric_factor'] = (
        0.4 * data['core_directional_efficiency'].fillna(0) +
        0.3 * data['intraday_regime_alignment'].fillna(0) +
        0.3 * data['multi_scale_confirmation'].fillna(0)
    )
    
    # Risk and Regime Adjustments
    # Volatility scaling
    data['volatility_5d'] = data['true_range'].rolling(window=5, min_periods=3).std()
    data['volatility_scaling'] = 1 / (data['volatility_5d'] + 1e-8)
    
    # Liquidity state filter
    data['liquidity_filter'] = np.where(data['volume'] > data['volume_ma_5'] * 0.5, 1, 0.5)
    
    # Persistence decay
    data['persistence_decay'] = np.exp(-0.1 * data['regime_persistence'].fillna(0))
    
    # Final factor with adjustments
    data['final_factor'] = (
        data['raw_asymmetric_factor'] *
        data['volatility_scaling'] *
        data['liquidity_filter'] *
        data['persistence_decay']
    )
    
    # Normalize for cross-sectional ranking
    data['factor_rank'] = data['final_factor'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x) >= 10 else np.nan
    )
    
    # Return the final factor series
    return data['factor_rank']
