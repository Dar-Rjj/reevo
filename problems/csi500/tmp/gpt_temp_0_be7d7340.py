import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Reversal with Liquidity Absorption Factor
    Combines reversal patterns with liquidity dynamics to predict short-term price reversals
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Reversal Patterns
    # Midday Price Rejection
    midpoint = (data['high'] + data['low']) / 2
    midpoint_return = (midpoint - data['open']) / data['open']
    closing_bias = (data['close'] - midpoint) / ((data['high'] - data['low']) / 2 + 1e-8)
    
    # Range Efficiency
    range_utilization = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    price_compression = 1 - range_utilization
    
    # Failed Breakout Signals
    rolling_high = data['high'].rolling(window=3, min_periods=1).max()
    rolling_low = data['low'].rolling(window=3, min_periods=1).min()
    
    high_touch = (data['high'] - rolling_high) / (data['high'] - data['low'] + 1e-8)
    low_touch = (rolling_low - data['low']) / (data['high'] - data['low'] + 1e-8)
    touch_retreat = (high_touch + low_touch) / 2
    
    # Momentum Exhaustion
    morning_momentum = (midpoint - data['open']) / data['open']
    afternoon_momentum = (data['close'] - midpoint) / midpoint
    momentum_deceleration = morning_momentum - afternoon_momentum
    
    # Volume decay (using rolling correlation as proxy)
    volume_rolling = data['volume'].rolling(window=5, min_periods=1)
    volume_trend = volume_rolling.apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0, raw=False)
    
    # 2. Liquidity Absorption Dynamics
    # Volume concentration proxy (using amount/volume relationship)
    large_trade_ratio = data['amount'] / (data['volume'] * data['close'] + 1e-8)
    
    # Price impact efficiency
    price_range = data['high'] - data['low']
    volume_efficiency = price_range / (data['volume'] + 1e-8)
    
    # Order flow imbalance proxy
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    buying_pressure = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    selling_pressure = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    order_imbalance = buying_pressure - selling_pressure
    
    # Liquidity exhaustion signals
    volume_spike = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    price_stability = 1 - (abs(data['close'] - data['open']) / data['open'])
    liquidity_exhaustion = volume_spike * price_stability
    
    # 3. Multi-Timeframe Microstructure
    # Opening vs closing dynamics
    opening_range = (data['high'].rolling(window=5, min_periods=1).max() - 
                    data['low'].rolling(window=5, min_periods=1).min())
    current_range = data['high'] - data['low']
    range_break_failure = opening_range / (current_range + 1e-8)
    
    # Volatility regimes
    intraday_volatility = (data['high'] - data['low']) / data['open']
    vol_regime = intraday_volatility / intraday_volatility.rolling(window=10, min_periods=1).mean()
    
    # Price continuity and mean reversion
    gap = data['open'] - data['close'].shift(1)
    gap_fill = -gap / (data['high'] - data['low'] + 1e-8)
    
    # 4. Synthesize Composite Factor
    # Reversal strength components
    reversal_strength = (
        -closing_bias * 0.3 +           # Negative closing bias suggests reversal
        price_compression * 0.2 +       # Price compression suggests exhaustion
        -touch_retreat * 0.15 +         # Failed breakouts
        -momentum_deceleration * 0.15 + # Momentum exhaustion
        -volume_trend * 0.2             # Volume decay
    )
    
    # Liquidity quality components
    liquidity_quality = (
        -large_trade_ratio * 0.3 +      # High large trades suggest poor liquidity
        -volume_efficiency * 0.25 +     # Low efficiency suggests absorption
        order_imbalance * 0.2 +         # Order flow imbalance
        -liquidity_exhaustion * 0.25    # Volume spikes without movement
    )
    
    # Microstructure context
    microstructure_context = (
        -range_break_failure * 0.4 +    # Failed range breaks
        vol_regime * 0.3 +              # Volatility regime adjustment
        gap_fill * 0.3                  # Gap fill tendency
    )
    
    # Final composite factor
    factor = (
        reversal_strength * 0.4 +
        liquidity_quality * 0.35 +
        microstructure_context * 0.25
    )
    
    # Apply dynamic normalization
    rolling_mean = factor.rolling(window=20, min_periods=1).mean()
    rolling_std = factor.rolling(window=20, min_periods=1).std()
    normalized_factor = (factor - rolling_mean) / (rolling_std + 1e-8)
    
    return normalized_factor
