import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Compute Volatility-Adjusted Intraday Momentum
    # Calculate Normalized Intraday Return
    raw_intraday_return = (data['close'] - data['open']) / (data['open'] + 1e-12)
    daily_range = data['high'] - data['low']
    normalized_intraday_return = raw_intraday_return * (daily_range / (data['open'] + 1e-12))
    
    # Calculate Volatility Persistence Component
    daily_volatility = daily_range / (data['open'] + 1e-12)
    volatility_momentum = daily_volatility / daily_volatility.shift(1)
    volatility_persistence = daily_volatility * volatility_momentum
    
    # Combine Momentum and Volatility Components
    momentum_component = normalized_intraday_return * volatility_persistence
    # Apply 3-day exponential weighting
    momentum_signal = momentum_component.ewm(span=3, adjust=False).mean()
    
    # Assess Price-Volume Efficiency Dynamics
    # Calculate Volume-Price Alignment Component
    volume_intensity = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    price_movement_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-12)
    volume_price_alignment = volume_intensity * price_movement_efficiency
    
    # Calculate Efficiency Persistence
    intraday_efficiency_trend = price_movement_efficiency / price_movement_efficiency.shift(1)
    volume_trend_consistency = data['volume'].rolling(window=5, min_periods=1).std() / data['volume'].rolling(window=10, min_periods=1).std()
    efficiency_persistence = intraday_efficiency_trend * volume_trend_consistency
    
    # Combine Efficiency Components
    efficiency_component = volume_price_alignment * efficiency_persistence
    # Apply directional confirmation with volatility adjustment
    efficiency_signal = efficiency_component * np.sign(price_movement_efficiency) * (1 + daily_volatility)
    
    # Generate Final Factor Signal
    # Synthesize Momentum and Efficiency Components
    combined_signal = momentum_signal * efficiency_signal
    # Enhanced when momentum direction aligns with efficiency improvement
    alignment_enhancement = np.sign(momentum_signal) * np.sign(efficiency_persistence)
    enhanced_signal = combined_signal * (1 + 0.5 * alignment_enhancement)
    
    # Apply Volatility-Based Signal Filter
    volatility_persistence_rolling = volatility_persistence.rolling(window=10, min_periods=1).apply(
        lambda x: (x > x.quantile(0.7)).mean() if len(x) > 0 else 0
    )
    volatility_filter = (volatility_persistence_rolling > 0.6).astype(float)
    
    # Final factor with volatility filtering
    final_factor = enhanced_signal * volatility_filter
    
    return final_factor
