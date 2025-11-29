import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Efficiency Calculation
    prev_close = data['close'].shift(1)
    true_range = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - prev_close),
            np.abs(data['low'] - prev_close)
        )
    )
    abs_price_change = np.abs(data['close'] - data['open'])
    efficiency_ratio = abs_price_change / (true_range + 1e-8)
    
    # Momentum Assessment
    gap_momentum = (data['open'] - prev_close) / (prev_close + 1e-8)
    range_momentum = (data['close'] - data['low']) / ((data['high'] - data['low']) + 1e-8)
    momentum_3day = data['close'] / data['close'].shift(3) - 1
    
    # Efficiency-Momentum Integration
    eff_weighted_gap_momentum = gap_momentum * efficiency_ratio
    eff_weighted_range_momentum = range_momentum * efficiency_ratio
    combined_efficiency_momentum = 0.4 * eff_weighted_gap_momentum + 0.4 * eff_weighted_range_momentum + 0.2 * momentum_3day
    
    # Liquidity Acceleration Dynamics
    daily_turnover = data['volume'] * data['close']
    turnover_ma_5day = daily_turnover.rolling(window=5, min_periods=3).mean()
    turnover_velocity = daily_turnover / (turnover_ma_5day + 1e-8)
    
    volume_ratio = data['volume'] / (data['volume'].shift(1) + 1e-8)
    turnover_change_3day = daily_turnover / daily_turnover.shift(3) - 1
    volume_ma_3day = data['volume'].rolling(window=3, min_periods=2).mean()
    volume_acceleration = data['volume'] / (volume_ma_3day + 1e-8)
    
    combined_liquidity_signal = 0.5 * turnover_velocity + 0.3 * volume_acceleration + 0.2 * turnover_change_3day
    
    # Cross-Sectional Momentum Alignment
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    rank_eff_gap_momentum = cross_sectional_rank(eff_weighted_gap_momentum)
    rank_eff_range_momentum = cross_sectional_rank(eff_weighted_range_momentum)
    rank_combined_eff_momentum = cross_sectional_rank(combined_efficiency_momentum)
    
    rank_turnover_velocity = cross_sectional_rank(turnover_velocity)
    rank_volume_acceleration = cross_sectional_rank(volume_acceleration)
    rank_combined_liquidity = cross_sectional_rank(combined_liquidity_signal)
    
    # Alignment Detection
    momentum_liquidity_alignment = (
        rank_combined_eff_momentum - rank_combined_liquidity
    ).abs()
    
    # Volatility Context Integration
    current_range = data['high'] - data['low']
    prev_range = data['high'].shift(1) - data['low'].shift(1)
    range_expansion = current_range / (prev_range + 1e-8) - 1
    
    range_momentum_trend = current_range / (data['high'].shift(3) - data['low'].shift(3) + 1e-8) - 1
    
    # Volatility regime classification
    volatility_regime = pd.cut(
        efficiency_ratio, 
        bins=[0, 0.3, 0.7, 1], 
        labels=[0, 1, 2]
    ).astype(float)
    
    # Efficiency-Volatility Interaction
    high_eff_range_expansion = efficiency_ratio * range_expansion
    high_eff_range_compression = efficiency_ratio * (-range_expansion)
    
    # Alpha Signal Synthesis
    # Signal Strength Calibration
    momentum_magnitude = combined_efficiency_momentum / (current_range / data['close'] + 1e-8)
    liquidity_efficiency_scaling = combined_liquidity_signal * efficiency_ratio
    
    # Volatility-adjusted weighting
    volatility_weight = 1 / (current_range.rolling(window=10, min_periods=5).std() + 1e-8)
    
    # Directional Signal Generation
    upward_acceleration = (
        (rank_combined_eff_momentum > 0.6) & 
        (rank_combined_liquidity > 0.6) & 
        (momentum_liquidity_alignment < 0.3)
    ).astype(float)
    
    downward_acceleration = (
        (rank_combined_eff_momentum < 0.4) & 
        (rank_combined_liquidity < 0.4) & 
        (momentum_liquidity_alignment < 0.3)
    ).astype(float)
    
    mean_reversion_timing = (
        (efficiency_ratio > 0.8) & (combined_efficiency_momentum.abs() > 0.02)
    ).astype(float) * -np.sign(combined_efficiency_momentum)
    
    # Final Alpha Construction
    # Combine ranked components with volatility adjustment
    momentum_component = (
        0.3 * rank_eff_gap_momentum + 
        0.3 * rank_eff_range_momentum + 
        0.4 * rank_combined_eff_momentum
    )
    
    liquidity_component = (
        0.5 * rank_turnover_velocity + 
        0.5 * rank_volume_acceleration
    )
    
    alignment_component = 1 / (momentum_liquidity_alignment + 0.1)
    
    volatility_component = volatility_weight * (
        0.4 * high_eff_range_expansion + 
        0.3 * high_eff_range_compression + 
        0.3 * volatility_regime
    )
    
    # Combine all components
    raw_alpha = (
        0.4 * momentum_component + 
        0.3 * liquidity_component + 
        0.2 * alignment_component + 
        0.1 * volatility_component
    )
    
    # Apply directional signals
    directional_adjustment = (
        upward_acceleration * 0.2 + 
        downward_acceleration * -0.2 + 
        mean_reversion_timing * 0.1
    )
    
    final_alpha = raw_alpha + directional_adjustment
    
    # Apply non-linear transformation to extreme values
    def winsorize_and_scale(series):
        q_low = series.quantile(0.05)
        q_high = series.quantile(0.95)
        scaled = np.where(
            series < q_low, q_low,
            np.where(series > q_high, q_high, series)
        )
        return (scaled - scaled.mean()) / (scaled.std() + 1e-8)
    
    # Final cross-sectional normalization
    alpha_factor = final_alpha.groupby(final_alpha.index).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    return alpha_factor
