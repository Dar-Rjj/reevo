import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Elasticity Momentum Framework
    Generates alpha factors based on elasticity, efficiency, and absorption patterns
    """
    data = df.copy()
    
    # Calculate daily ranges
    data['daily_range'] = data['high'] - data['low']
    data['morning_range'] = data['high'] - data['open']
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Short-Term Elasticity Analysis
    data['5d_price_change'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['5d_avg_range'] = data['daily_range'].rolling(window=5).mean()
    data['5d_elasticity'] = np.abs(data['5d_price_change']) / data['5d_avg_range']
    data['5d_elasticity_momentum'] = data['5d_elasticity'] - data['5d_elasticity'].shift(5)
    
    # Gap elasticity
    data['gap_elasticity'] = np.abs(data['overnight_gap']) / data['morning_range']
    
    # Medium-Term Elasticity Framework
    data['20d_price_change'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['20d_avg_range'] = data['daily_range'].rolling(window=20).mean()
    data['20d_elasticity'] = np.abs(data['20d_price_change']) / data['20d_avg_range']
    data['elasticity_divergence'] = data['5d_elasticity'] - data['20d_elasticity']
    data['divergence_persistence'] = data['elasticity_divergence'].rolling(window=5).mean()
    
    # Efficiency-Elasticity Interaction Matrix
    # Morning session efficiency
    morning_close = (data['open'] + data['high']) / 2  # Proxy for morning close
    data['morning_efficiency'] = (morning_close - data['open']) / (data['high'] - data['low'])
    
    # Afternoon session efficiency
    afternoon_open = (data['low'] + data['close']) / 2  # Proxy for afternoon open
    data['afternoon_efficiency'] = (data['close'] - afternoon_open) / (data['high'] - data['low'])
    
    # Cross-session efficiency correlation
    data['efficiency_correlation'] = data['morning_efficiency'].rolling(window=5).corr(data['afternoon_efficiency'])
    
    # Elasticity Regime Detection
    data['10d_elasticity_vol'] = data['5d_elasticity'].rolling(window=10).std()
    data['elasticity_momentum_10d'] = data['5d_elasticity'] - data['5d_elasticity'].shift(10)
    
    # Regime classification
    high_elasticity = data['5d_elasticity'] > data['5d_elasticity'].rolling(window=20).quantile(0.7)
    stable_elasticity = data['10d_elasticity_vol'] < data['10d_elasticity_vol'].rolling(window=20).quantile(0.3)
    data['elasticity_regime'] = np.where(high_elasticity & stable_elasticity, 2, 
                                       np.where(high_elasticity, 1, 0))
    
    # Gap-Enhanced Order Flow Patterns
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['5d_pressure_accum'] = data['intraday_pressure'].rolling(window=5).sum()
    
    # Volume absorption analysis
    data['volume_intensity'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['gap_absorption'] = data['volume_intensity'] * np.abs(data['overnight_gap'])
    
    # Cross-Scale Elasticity-Efficiency Alignment
    data['efficiency_elasticity_alignment'] = (data['morning_efficiency'] * data['afternoon_efficiency']) / (data['5d_elasticity'] + 1e-6)
    
    # Quality momentum signals
    high_efficiency = (data['morning_efficiency'] > 0.6) & (data['afternoon_efficiency'] > 0.6)
    low_elasticity = data['5d_elasticity'] < data['5d_elasticity'].rolling(window=20).quantile(0.3)
    data['quality_momentum'] = np.where(high_efficiency & low_elasticity, 1, 0)
    
    # Range-Elasticity Memory Framework
    data['price_response_volume'] = np.abs(data['close'].pct_change()) / (data['volume'] + 1e-6)
    
    # Previous session level analysis
    data['prev_high_elasticity'] = data['high'].shift(1).rolling(window=5).apply(
        lambda x: np.mean(np.abs((x - x.shift(1)) / x.shift(1)) / data['daily_range'].shift(1).rolling(window=5).mean())
    )
    
    # Regime-Adaptive Elasticity Integration
    # Elasticity-weighted momentum
    low_elasticity_mask = data['5d_elasticity'] < data['5d_elasticity'].rolling(window=20).quantile(0.3)
    high_elasticity_mask = data['5d_elasticity'] > data['5d_elasticity'].rolling(window=20).quantile(0.7)
    
    data['elasticity_weighted_momentum'] = np.where(
        low_elasticity_mask,
        data['5d_price_change'] * (1 + data['morning_efficiency']),
        np.where(
            high_elasticity_mask,
            data['5d_price_change'] * data['morning_efficiency'],
            data['5d_price_change']
        )
    )
    
    # Efficiency-confirmed elasticity
    high_volume_efficiency = data['volume_intensity'] > 1.2
    strong_elasticity = data['5d_elasticity'] > data['5d_elasticity'].rolling(window=20).quantile(0.7)
    
    data['efficiency_confirmed_trend'] = np.where(
        high_volume_efficiency & strong_elasticity,
        data['5d_price_change'] * data['morning_efficiency'],
        0
    )
    
    # Composite Elasticity Alpha Generation
    # Multi-dimensional confirmation
    momentum_alignment = np.sign(data['5d_price_change']) == np.sign(data['intraday_pressure'])
    efficiency_alignment = np.sign(data['morning_efficiency']) == np.sign(data['afternoon_efficiency'])
    
    data['multi_dim_confirmation'] = np.where(
        momentum_alignment & efficiency_alignment,
        data['5d_price_change'] * data['morning_efficiency'] * data['afternoon_efficiency'],
        data['5d_price_change'] * 0.5
    )
    
    # Dynamic factor weighting based on elasticity regime
    low_elasticity_weight = np.where(low_elasticity_mask, 0.7, 0.3)
    high_elasticity_weight = np.where(high_elasticity_mask, 0.6, 0.4)
    
    # Final composite alpha factor
    momentum_component = data['elasticity_weighted_momentum'] * low_elasticity_weight
    efficiency_component = data['efficiency_confirmed_trend'] * high_elasticity_weight
    absorption_component = data['gap_absorption'] * data['intraday_pressure']
    
    data['composite_elasticity_alpha'] = (
        momentum_component * 0.4 +
        efficiency_component * 0.35 +
        absorption_component * 0.25 +
        data['multi_dim_confirmation'] * 0.1
    )
    
    # Signal validation and regime persistence
    regime_stability = data['elasticity_regime'].rolling(window=5).std() == 0
    data['validated_alpha'] = np.where(
        regime_stability,
        data['composite_elasticity_alpha'] * 1.2,
        data['composite_elasticity_alpha'] * 0.8
    )
    
    return data['validated_alpha']
