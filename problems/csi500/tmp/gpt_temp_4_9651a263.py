import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Momentum Acceleration Component
    # Raw Momentum
    raw_momentum = (data['close'] - data['open']) / (data['open'] + 1e-12)
    
    # Momentum Change
    momentum_change = raw_momentum / raw_momentum.shift(1)
    momentum_change = momentum_change.replace([np.inf, -np.inf], np.nan)
    
    # Acceleration
    acceleration = momentum_change / momentum_change.shift(1)
    acceleration = acceleration.replace([np.inf, -np.inf], np.nan)
    
    # Volatility-Adjusted Acceleration
    intraday_volatility = (data['high'] - data['low']) / (data['open'] + 1e-12)
    adjusted_acceleration = acceleration * intraday_volatility
    
    # Momentum Persistence Filter
    momentum_direction = np.sign(raw_momentum)
    momentum_consistency = momentum_direction.rolling(window=5).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) == 5 else np.nan
    )
    filtered_acceleration = adjusted_acceleration * momentum_consistency
    
    # Price-Volume Divergence Dynamics
    # Volume-Price Divergence Component
    normalized_volume = data['volume'] / data['volume'].rolling(window=20).mean()
    price_range_utilization = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-12)
    divergence = normalized_volume - price_range_utilization
    
    # Divergence Persistence
    divergence_trend = divergence / divergence.shift(1)
    divergence_trend = divergence_trend.replace([np.inf, -np.inf], np.nan)
    
    volume_stability = data['volume'].rolling(window=5).std() / data['volume'].rolling(window=20).std()
    persistence = divergence_trend * volume_stability
    
    # Combine Divergence Components
    volatility_weight = 1 / (intraday_volatility.rolling(window=10).std() + 1e-12)
    combined_divergence = divergence * persistence * volatility_weight
    
    # Generate Final Factor Signal
    # Synthesize Acceleration and Divergence Components
    raw_signal = filtered_acceleration * combined_divergence
    
    # Enhanced directional alignment
    directional_alignment = np.sign(filtered_acceleration) * np.sign(combined_divergence)
    enhanced_signal = raw_signal * (1 + 0.5 * directional_alignment)
    
    # Volume-Based Signal Validation
    volume_percentile = data['volume'].rolling(window=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x) == 10 else np.nan
    )
    
    # Final factor with volume confirmation
    final_factor = enhanced_signal * (volume_percentile > 0.3)
    
    return final_factor
