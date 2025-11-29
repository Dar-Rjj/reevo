import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Price Acceleration with Volatility Adjustment
    # Calculate intraday momentum
    intraday_momentum = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_momentum = intraday_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Volatility adjustment using 10-day rolling standard deviation of Close
    vol_adj = data['close'].rolling(window=10, min_periods=5).std()
    volatility_adjusted_acceleration = intraday_momentum / vol_adj
    
    # 2. Liquidity-Adjusted Volume Pressure
    # Calculate average trade size
    avg_trade_size = data['amount'] / data['volume']
    avg_trade_size = avg_trade_size.replace([np.inf, -np.inf], np.nan)
    
    # Compute volume pressure
    volume_pressure = data['volume'] * (data['close'] - data['open']) / data['close']
    
    # Filter by liquidity regime (trade size > 20-day rolling median)
    trade_size_median = avg_trade_size.rolling(window=20, min_periods=10).median()
    liquidity_filter = (avg_trade_size > trade_size_median).astype(float)
    liquidity_adjusted_pressure = volume_pressure * liquidity_filter
    
    # 3. Combine Acceleration and Pressure with Regime Detection
    combined_signal = volatility_adjusted_acceleration * liquidity_adjusted_pressure
    
    # High-volatility regime detection
    daily_range = (data['high'] - data['low']) / data['close']
    range_percentile = daily_range.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) > 0 else False
    )
    high_vol_regime = range_percentile.astype(float) * 1.5 + (1 - range_percentile.astype(float))
    regime_amplified_signal = combined_signal * high_vol_regime
    
    # 4. Price-Level Anchoring Effect
    # Calculate recent price extremes
    recent_high = data['high'].rolling(window=5, min_periods=3).max()
    recent_low = data['low'].rolling(window=5, min_periods=3).min()
    
    # Compute distances to extremes
    dist_to_high = (recent_high - data['close']) / recent_high
    dist_to_low = (data['close'] - recent_low) / data['close']
    
    # Anchoring adjustment - reduce magnitude near highs, increase near lows
    high_anchor_adj = 1 - (dist_to_high / (dist_to_high + dist_to_low))
    low_anchor_adj = dist_to_low / (dist_to_high + dist_to_low)
    anchoring_adjustment = low_anchor_adj - high_anchor_adj
    
    # Apply anchoring adjustment
    final_factor = regime_amplified_signal * (1 + anchoring_adjustment)
    
    return final_factor
