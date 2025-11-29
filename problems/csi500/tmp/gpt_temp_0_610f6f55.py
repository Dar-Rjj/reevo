import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Efficiency Momentum
    # Daily efficiency: (Close - Open) / (High - Low)
    daily_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    daily_efficiency = daily_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # 3-day efficiency momentum
    eff_3day_avg = daily_efficiency.rolling(window=3, min_periods=1).mean()
    efficiency_momentum = daily_efficiency - eff_3day_avg
    
    # Volume-weighted efficiency
    volume_5day_avg = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_acceleration = data['volume'] / volume_5day_avg
    volume_weighted_efficiency = daily_efficiency * volume_acceleration
    
    # Volume-efficiency correlation consistency (5-day rolling correlation)
    eff_vol_corr = daily_efficiency.rolling(window=5, min_periods=1).corr(volume_acceleration)
    
    # 2. Transaction Intensity Regime Analysis
    # Transaction size: Amount / Volume
    transaction_size = data['amount'] / data['volume']
    transaction_size = transaction_size.replace([np.inf, -np.inf], np.nan)
    
    # Transaction size momentum
    transaction_momentum = transaction_size / transaction_size.shift(1)
    transaction_momentum = transaction_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Directional price persistence strength
    price_direction = np.sign(data['close'] - data['open'])
    directional_persistence = price_direction.rolling(window=3, min_periods=1).sum() / 3
    
    # Regime-weighted intensity
    regime_intensity = transaction_momentum * directional_persistence
    
    # 3. Volatility Transition Efficiency
    # Range ratio: (High - Low) / Previous (High - Low)
    daily_range = data['high'] - data['low']
    range_ratio = daily_range / daily_range.shift(1)
    range_ratio = range_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Volatility regime detection using ratio sequences
    vol_regime = range_ratio.rolling(window=3, min_periods=1).apply(
        lambda x: 1 if (x > 1.1).sum() >= 2 else (-1 if (x < 0.9).sum() >= 2 else 0)
    )
    
    # Efficiency during transitions
    transition_efficiency = daily_efficiency * vol_regime
    
    # 4. Multi-Scale Pressure Dynamics
    # Intraday pressure gradient (same as daily efficiency)
    pressure_gradient = daily_efficiency
    
    # Inter-day pressure persistence
    interday_pressure = daily_efficiency.rolling(window=5, min_periods=1).std()
    
    # Gradient-persistence alignment
    mean_reversion_factor = np.where(
        (pressure_gradient.abs() > pressure_gradient.rolling(window=10, min_periods=1).quantile(0.7)) & 
        (interday_pressure < interday_pressure.rolling(window=10, min_periods=1).quantile(0.3)),
        -pressure_gradient, 0
    )
    
    trend_factor = np.where(
        (pressure_gradient.abs() < pressure_gradient.rolling(window=10, min_periods=1).quantile(0.3)) & 
        (interday_pressure > interday_pressure.rolling(window=10, min_periods=1).quantile(0.7)),
        pressure_gradient, 0
    )
    
    # 5. Liquidity Barrier Quality Assessment
    # Detect price levels with volume concentration (using rolling percentiles)
    high_volume_levels = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: 1 if x.iloc[-1] > np.percentile(x, 80) else 0
    )
    
    # Amount acceleration during barrier tests
    amount_5day_avg = data['amount'].rolling(window=5, min_periods=1).mean()
    amount_acceleration = data['amount'] / amount_5day_avg
    
    # Barrier strength
    barrier_strength = high_volume_levels * amount_acceleration
    
    # Breakout validation
    price_change = (data['close'] - data['open']) / data['open']
    breakout_signal = np.where(
        (price_change.abs() > price_change.rolling(window=10, min_periods=1).std()) & 
        (amount_acceleration > 1.5),
        price_change, 0
    )
    
    # 6. Temporal Alignment Momentum
    # Price-volume lead-lag analysis (5-day correlation)
    price_volume_corr = data['close'].pct_change().rolling(window=5, min_periods=1).corr(
        data['volume'].pct_change()
    )
    
    # Alignment quality scoring
    price_lead_momentum = np.where(price_volume_corr > 0.3, 1, 0)
    volume_lead_momentum = np.where(price_volume_corr < -0.3, 1, 0)
    sync_momentum = np.where((price_volume_corr >= -0.1) & (price_volume_corr <= 0.1), 1, 0)
    
    # 7. Microstructure Gap Quality
    # Opening gap: (Open - Previous Close) / Previous Close
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap persistence through trading session
    gap_fill_ratio = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1))
    gap_fill_ratio = gap_fill_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Gap classification
    noise_gap = np.where(
        (gap_fill_ratio.abs() > 0.8) & 
        (data['volume'] < data['volume'].rolling(window=10, min_periods=1).quantile(0.3)),
        -opening_gap, 0
    )
    
    fundamental_gap = np.where(
        (gap_fill_ratio.abs() < 0.2) & 
        (data['volume'] > data['volume'].rolling(window=10, min_periods=1).quantile(0.7)),
        opening_gap, 0
    )
    
    # 8. Range Expansion Quality Momentum
    # Range expansion relative to recent volatility
    range_expansion = daily_range / daily_range.rolling(window=10, min_periods=1).mean()
    
    # Expansion quality scoring
    efficient_expansion = np.where(
        (range_expansion > 1.2) & 
        (daily_efficiency.abs() > daily_efficiency.rolling(window=10, min_periods=1).quantile(0.7)),
        daily_efficiency, 0
    )
    
    inefficient_expansion = np.where(
        (range_expansion > 1.2) & 
        (daily_efficiency.abs() < daily_efficiency.rolling(window=10, min_periods=1).quantile(0.3)),
        -daily_efficiency, 0
    )
    
    volume_confirmed_expansion = np.where(
        (range_expansion > 1.2) & 
        (volume_acceleration > 1.5),
        daily_efficiency, 0
    )
    
    # Combine factors with appropriate weights
    factor_components = {
        'efficiency_momentum': efficiency_momentum * 0.15,
        'volume_weighted_efficiency': volume_weighted_efficiency * 0.12,
        'regime_intensity': regime_intensity * 0.10,
        'transition_efficiency': transition_efficiency * 0.08,
        'mean_reversion': mean_reversion_factor * 0.10,
        'trend': trend_factor * 0.08,
        'breakout_signal': breakout_signal * 0.12,
        'sync_momentum': sync_momentum * 0.05,
        'fundamental_gap': fundamental_gap * 0.08,
        'efficient_expansion': efficient_expansion * 0.12
    }
    
    # Calculate final factor value
    final_factor = pd.Series(0, index=data.index)
    for component, weight in factor_components.items():
        final_factor += weight
    
    # Normalize the factor
    final_factor = (final_factor - final_factor.rolling(window=20, min_periods=1).mean()) / \
                   final_factor.rolling(window=20, min_periods=1).std()
    
    return final_factor
