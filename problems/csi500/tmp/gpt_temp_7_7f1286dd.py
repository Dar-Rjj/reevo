import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional momentum efficiency factor based on multi-scale efficiency dynamics,
    volume-efficiency convergence, gap absorption patterns, and microstructural friction signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Scale Efficiency Dynamics
    # Daily efficiency: (Close - Open) / (High - Low)
    daily_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    daily_efficiency = daily_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Short-term efficiency (3-day rolling average)
    short_term_efficiency = daily_efficiency.rolling(window=3, min_periods=2).mean()
    
    # Medium-term efficiency (10-day rolling average)
    medium_term_efficiency = daily_efficiency.rolling(window=10, min_periods=5).mean()
    
    # Efficiency divergence: Short-term minus Medium-term
    efficiency_divergence = short_term_efficiency - medium_term_efficiency
    
    # Efficiency momentum: Rate of efficiency change
    efficiency_momentum = daily_efficiency.diff(3) / daily_efficiency.rolling(window=5).std()
    
    # Efficiency regime classification
    price_range = data['high'] - data['low']
    range_change = price_range.pct_change(3)
    
    # Improving efficiency + expanding range → quality momentum
    quality_momentum = ((efficiency_divergence > 0) & (range_change > 0)).astype(float)
    
    # Deteriorating efficiency + contracting range → exhaustion
    exhaustion = ((efficiency_divergence < 0) & (range_change < 0)).astype(float)
    
    # Stable efficiency + volatile range → contested moves
    efficiency_volatility = daily_efficiency.rolling(window=5).std()
    contested_moves = ((efficiency_divergence.abs() < 0.1) & (range_change.abs() > 0.05)).astype(float)
    
    # 2. Volume-Efficiency Convergence
    # Volume concentration analysis
    # Using daily volume patterns as proxy for intraday concentration
    volume_persistence = data['volume'].rolling(window=3).apply(
        lambda x: 1 if (x.diff().fillna(0) > 0).all() else (-1 if (x.diff().fillna(0) < 0).all() else 0)
    )
    
    # Efficiency-Volume Alignment
    volume_change = data['volume'].pct_change(3)
    
    # High efficiency + concentrated volume → institutional program
    institutional_program = ((daily_efficiency > daily_efficiency.rolling(10).mean()) & 
                            (volume_persistence > 0)).astype(float)
    
    # Low efficiency + dispersed volume → noise trading
    noise_trading = ((daily_efficiency < daily_efficiency.rolling(10).mean()) & 
                    (volume_persistence < 0)).astype(float)
    
    # Efficiency improvement + volume expansion → breakout confirmation
    breakout_confirmation = ((efficiency_momentum > 0) & (volume_change > 0)).astype(float)
    
    # Efficiency decline + volume contraction → trend weakness
    trend_weakness = ((efficiency_momentum < 0) & (volume_change < 0)).astype(float)
    
    # 3. Gap Absorption Momentum
    # Opening Gap Dynamics
    gap_magnitude = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Intraday Absorption Patterns
    gap_filled_up = ((gap_magnitude > 0) & (data['low'] <= data['close'].shift(1))).astype(float)
    gap_filled_down = ((gap_magnitude < 0) & (data['high'] >= data['close'].shift(1))).astype(float)
    
    # Gap absorption efficiency
    gap_absorption_efficiency = np.where(
        gap_magnitude > 0,
        (data['close'].shift(1) - data['low']) / gap_magnitude.abs(),
        np.where(
            gap_magnitude < 0,
            (data['high'] - data['close'].shift(1)) / gap_magnitude.abs(),
            0
        )
    )
    
    # Volume absorption at gap levels
    gap_volume_ratio = data['volume'] / data['volume'].rolling(window=10).mean()
    absorption_score = gap_absorption_efficiency * gap_volume_ratio
    
    # 4. Microstructural Friction Signals
    # Transaction Cost Proxy using High-Low range normalized by price
    friction_proxy = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    # Friction persistence
    friction_persistence = friction_proxy.rolling(window=5).std()
    
    # Friction-Momentum Interaction
    # High friction + efficiency breakout → forced institutional flow
    forced_institutional = ((friction_proxy > friction_proxy.rolling(10).mean()) & 
                           (efficiency_momentum > 0)).astype(float)
    
    # Low friction + efficiency decline → lack of conviction
    lack_conviction = ((friction_proxy < friction_proxy.rolling(10).mean()) & 
                      (efficiency_momentum < 0)).astype(float)
    
    # Progressive friction reduction → accumulation completion
    friction_reduction = (friction_proxy.rolling(window=3).mean() < 
                         friction_proxy.rolling(window=10).mean()).astype(float)
    
    # 5. Cross-Timeframe Momentum Alignment
    # Multi-Scale Convergence
    aligned_efficiency = ((short_term_efficiency > medium_term_efficiency) & 
                         (medium_term_efficiency > medium_term_efficiency.shift(5))).astype(float)
    
    contradictory_signals = ((short_term_efficiency * medium_term_efficiency) < 0).astype(float)
    
    # 6. Composite Alpha Signal
    # Factor Integration with equal weights for demonstration
    composite_signal = (
        quality_momentum * 0.15 +
        institutional_program * 0.15 +
        breakout_confirmation * 0.15 +
        absorption_score.fillna(0) * 0.15 +
        forced_institutional * 0.10 +
        friction_reduction * 0.10 +
        aligned_efficiency * 0.10 +
        -exhaustion * 0.05 +
        -noise_trading * 0.05
    )
    
    # Normalize the final signal
    final_signal = (composite_signal - composite_signal.rolling(window=20, min_periods=10).mean()) / \
                   composite_signal.rolling(window=20, min_periods=10).std()
    
    return final_signal
