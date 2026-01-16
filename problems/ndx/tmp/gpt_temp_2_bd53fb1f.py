import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # 1. Measure Price Efficiency
    # Intraday Price Ratio
    intraday_ratio = (data['high'] - data['low']) / data['close']
    
    # 2. Normalize Efficiency
    efficiency = intraday_ratio.rolling(window=30).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if len(x[:-1]) > 1 else 0
    )
    
    # 3. Apply Volume Adjustment
    # Calculate Volume Ratio (current volume / rolling mean volume)
    volume_ratio = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # Scale Efficiency by Volume
    volume_adjusted_efficiency = efficiency * volume_ratio
    
    # Normalize by Historical Range (30 days)
    normalized_efficiency = volume_adjusted_efficiency.rolling(window=30).apply(
        lambda x: (x[-1] - x[:-1].min()) / (x[:-1].max() - x[:-1].min()) if len(x[:-1]) > 1 else 0
    )
    
    # 4. Integrate Momentum Confirmation
    # Calculate Momentum Signal (15-day price change)
    momentum = data['close'] - data['close'].shift(15)
    
    # Adjust Final Factor
    factor = normalized_efficiency * momentum
    
    # Final normalization
    final_factor = factor.rolling(window=30).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if len(x[:-1]) > 1 else 0
    )
    
    return final_factor
