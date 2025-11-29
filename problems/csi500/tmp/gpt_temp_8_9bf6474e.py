import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Price Rejection Patterns
    high_rejection_strength = (df['high'] - df['close']) / (df['high'] - df['low'])
    low_rejection_strength = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Replace infinite values with NaN
    high_rejection_strength = high_rejection_strength.replace([float('inf'), -float('inf')], float('nan'))
    low_rejection_strength = low_rejection_strength.replace([float('inf'), -float('inf')], float('nan'))
    
    # Compute Volume Acceleration and Flow Patterns
    volume_acceleration = df['volume'] / df['volume'].rolling(5).mean()
    # Note: Early-late session ratio requires intraday data which is not available
    # Using daily volume ratio approximation
    early_late_ratio = df['volume'] / df['volume'].shift(1)
    
    # Assess Volatility Regime Context
    daily_range = df['high'] - df['low']
    volatility_regime_flag = daily_range > 1.5 * daily_range.rolling(10).median()
    
    # Generate Volume-Weighted Rejection Signals
    high_rejection_signal = high_rejection_strength * volume_acceleration
    low_rejection_signal = low_rejection_strength * volume_acceleration
    
    # Create Conditional Composite Factor
    # Determine dominant rejection direction
    dominant_signal = pd.Series(index=df.index, dtype=float)
    for date in df.index:
        high_signal = abs(high_rejection_signal.loc[date]) if not pd.isna(high_rejection_signal.loc[date]) else 0
        low_signal = abs(low_rejection_signal.loc[date]) if not pd.isna(low_rejection_signal.loc[date]) else 0
        
        if high_signal >= low_signal:
            dominant_signal.loc[date] = high_rejection_signal.loc[date]
        else:
            dominant_signal.loc[date] = low_rejection_signal.loc[date]
    
    # Apply volatility regime conditioning
    conditional_signal = dominant_signal.copy()
    for date in df.index:
        if volatility_regime_flag.loc[date] and not pd.isna(early_late_ratio.loc[date]):
            conditional_signal.loc[date] = dominant_signal.loc[date] * early_late_ratio.loc[date]
    
    # Scale by daily range context
    range_scaling = daily_range / daily_range.rolling(10).mean()
    final_factor = conditional_signal * range_scaling
    
    return final_factor
