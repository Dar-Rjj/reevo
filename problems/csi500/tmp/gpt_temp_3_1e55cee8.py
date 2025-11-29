import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range Components
    df['HL'] = df['high'] - df['low']
    df['HC'] = np.abs(df['high'] - df['close'].shift(1))
    df['LC'] = np.abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    
    # Volatility Regime Classification
    df['TR_percentile'] = df['TR'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    conditions = [
        df['TR_percentile'] > 0.8,
        (df['TR_percentile'] >= 0.2) & (df['TR_percentile'] <= 0.8),
        df['TR_percentile'] < 0.2
    ]
    choices = ['high', 'normal', 'low']
    df['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Regime-Specific Momentum Calculation
    regime_momentum = pd.Series(index=df.index, dtype=float)
    
    # High Volatility Regime
    high_mask = df['vol_regime'] == 'high'
    intraday_pressure = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_pressure = intraday_pressure.replace([np.inf, -np.inf], 0)
    regime_momentum[high_mask] = intraday_pressure[high_mask] * df['volume'][high_mask]
    
    # Normal Volatility Regime
    normal_mask = df['vol_regime'] == 'normal'
    price_continuation = df['close'] - df['close'].shift(1)
    amount_adjusted = price_continuation / df['amount']
    amount_adjusted = amount_adjusted.replace([np.inf, -np.inf], 0)
    regime_momentum[normal_mask] = amount_adjusted[normal_mask]
    
    # Low Volatility Regime
    low_mask = df['vol_regime'] == 'low'
    midpoint_deviation = (df['high'] + df['low']) / 2 - df['close'].shift(1)
    range_normalized = midpoint_deviation / (df['high'] - df['low'])
    range_normalized = range_normalized.replace([np.inf, -np.inf], 0)
    regime_momentum[low_mask] = range_normalized[low_mask]
    
    # Volume-Efficiency Enhancement
    # Volume Trend Strength
    volume_ma_5 = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_trend_strength = df['volume'] / volume_ma_5
    volume_acceleration = df['volume'] / df['volume'].shift(1)
    volume_acceleration = volume_acceleration.replace([np.inf, -np.inf], 1)
    
    # Trading Efficiency
    price_movement_efficiency = df['amount'] / (df['volume'] * df['TR'])
    price_movement_efficiency = price_movement_efficiency.replace([np.inf, -np.inf], 0)
    efficiency_multiplier = price_movement_efficiency * (df['close'] - df['open'])
    
    # Final Alpha Factor Construction
    # Combine Regime Momentum with Volume Dynamics
    intermediate_factor = regime_momentum * volume_trend_strength * volume_acceleration
    
    # Enhance with Efficiency Signal
    combined_signal = intermediate_factor * efficiency_multiplier
    
    # Final Factor transformation
    final_factor = np.sign(combined_signal) * np.sqrt(np.abs(combined_signal))
    
    return final_factor
