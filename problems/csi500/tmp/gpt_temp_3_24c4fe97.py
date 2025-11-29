import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Component
    # Calculate Intraday Momentum
    intraday_momentum = (data['close'] - data['low']) / (data['high'] - data['low'])
    intraday_momentum = intraday_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Momentum Persistence Analysis
    # Calculate rolling correlation between current and lagged intraday momentum
    momentum_corr = intraday_momentum.rolling(window=5, min_periods=3).corr(intraday_momentum.shift(1))
    
    # Volume Confirmation
    volume_avg_20 = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = data['volume'] / volume_avg_20
    volume_weighted_persistence = momentum_corr * volume_ratio
    
    # Price Range Efficiency Component
    # Compute Daily Price Range
    daily_range = data['high'] - data['low']
    
    # Assess Range Efficiency
    abs_return = abs(data['close'] - data['close'].shift(1))
    range_efficiency = abs_return / daily_range
    range_efficiency = range_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Volume Context Analysis
    volume_percentile = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] > x[:-1]).sum() / len(x[:-1]) if len(x[:-1]) > 0 else np.nan
    )
    efficiency_volume_combined = range_efficiency * volume_percentile
    
    # Dual Momentum Acceleration Component
    # Intraday Momentum Acceleration
    intraday_momentum_5d = intraday_momentum.rolling(window=5, min_periods=3).mean()
    intraday_momentum_10d = intraday_momentum.rolling(window=10, min_periods=5).mean()
    momentum_acceleration = intraday_momentum_5d - intraday_momentum_10d
    
    # Volatility-Adjusted Acceleration
    # True Range Calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Volatility Normalization
    rolling_volatility = true_range.rolling(window=20, min_periods=10).mean()
    volatility_adjusted_acceleration = momentum_acceleration / rolling_volatility
    volatility_adjusted_acceleration = volatility_adjusted_acceleration.replace([np.inf, -np.inf], np.nan)
    
    # Combine all components
    # Normalize each component by their rolling z-scores
    def rolling_zscore(series, window=20):
        return (series - series.rolling(window=window, min_periods=10).mean()) / series.rolling(window=window, min_periods=10).std()
    
    component1 = rolling_zscore(volume_weighted_persistence)
    component2 = rolling_zscore(efficiency_volume_combined)
    component3 = rolling_zscore(volatility_adjusted_acceleration)
    
    # Final factor: weighted combination
    final_factor = 0.4 * component1 + 0.35 * component2 + 0.25 * component3
    
    return final_factor
