import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate intraday efficiency
    intraday_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Smooth intraday efficiency with a 5-day EMA
    smoothed_efficiency = intraday_efficiency.ewm(span=5, adjust=False).mean()
    
    # Calculate volume percentile
    volume_percentile = data['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Normalize volume percentile to [0,1] range
    normalized_volume_percentile = (volume_percentile - volume_percentile.min()) / (volume_percentile.max() - volume_percentile.min())
    
    # Combine signals by multiplying efficiency by volume percentile
    combined_signal = smoothed_efficiency * normalized_volume_percentile
    
    # Scale by volatility adjustment (divide by 10-day std of close)
    volatility_adjustment = data['close'].rolling(window=10).std()
    final_signal = combined_signal / volatility_adjustment
    
    return final_signal
