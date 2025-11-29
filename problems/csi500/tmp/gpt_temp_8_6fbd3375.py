import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Pressure Component
    # Calculate intraday price pressure: (2 * Close - High - Low) / (High - Low)
    price_pressure = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
    price_pressure = price_pressure.replace([np.inf, -np.inf], np.nan)
    
    # Compute momentum persistence: current pressure × previous day pressure
    momentum_persistence = price_pressure * price_pressure.shift(1)
    
    # Apply pressure transformation: tanh(momentum persistence)
    momentum_pressure = np.tanh(momentum_persistence)
    
    # Volume Acceleration Component
    # Calculate volume velocity: Volume / (High - Low)
    volume_velocity = df['volume'] / (df['high'] - df['low'])
    volume_velocity = volume_velocity.replace([np.inf, -np.inf], np.nan)
    
    # Compute acceleration ratio: current velocity / 3-day mean velocity
    mean_velocity_3d = volume_velocity.rolling(window=3, min_periods=1).mean()
    acceleration_ratio = volume_velocity / mean_velocity_3d
    acceleration_ratio = acceleration_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Apply acceleration modifier: log(1 + abs(acceleration ratio - 1))
    volume_acceleration = np.log(1 + np.abs(acceleration_ratio - 1))
    
    # Regime Persistence Component
    # Calculate regime indicator: sign(Close - (High + Low)/2)
    regime_indicator = np.sign(df['close'] - (df['high'] + df['low']) / 2)
    
    # Compute regime strength: count of same sign in past 5 days
    regime_strength = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 5:
            regime_strength.iloc[i] = 1
        else:
            current_sign = regime_indicator.iloc[i]
            past_signs = regime_indicator.iloc[i-5:i]
            regime_strength.iloc[i] = (past_signs == current_sign).sum()
    
    # Apply regime multiplier: 1 + regime strength / 5
    regime_persistence = 1 + regime_strength / 5
    
    # Range Position Confirmation
    # Calculate normalized range position: (Close - Low) / (High - Low)
    range_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    range_position = range_position.replace([np.inf, -np.inf], np.nan)
    
    # Compute volume-weighted position: range position × Volume
    volume_weighted_position = range_position * df['volume']
    
    # Apply confirmation filter: 1 / (1 + exp(-5 * volume-weighted position))
    range_confirmation = 1 / (1 + np.exp(-5 * volume_weighted_position))
    
    # Final Factor Synthesis
    # Combine momentum pressure × volume acceleration × regime persistence
    combined_factor = momentum_pressure * volume_acceleration * regime_persistence
    
    # Multiply by range position confirmation
    combined_factor = combined_factor * range_confirmation
    
    # Apply momentum-weighted smoothing using 2-day pressure ratio
    pressure_ratio = price_pressure / price_pressure.shift(1)
    pressure_ratio = pressure_ratio.replace([np.inf, -np.inf], np.nan)
    momentum_weight = np.tanh(pressure_ratio)
    
    # Final factor with momentum-weighted smoothing
    final_factor = combined_factor * momentum_weight
    
    return final_factor
