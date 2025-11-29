import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Price Reversal Components
    # Morning Rejection Strength
    high_low_range = data['high'] - data['low']
    morning_rejection = (data['open'] - data['low']) / np.where(high_low_range != 0, high_low_range, 1)
    
    # Afternoon Recovery Strength
    afternoon_recovery = (data['close'] - data['low']) / np.where(high_low_range != 0, high_low_range, 1)
    
    # 2. Volatility Adjustment
    intraday_volatility = high_low_range
    vol_adjusted_morning = morning_rejection * intraday_volatility
    vol_adjusted_afternoon = afternoon_recovery * intraday_volatility
    
    # 3. Volume Divergence Analysis
    # Abnormal Volume Detection - 10-day rolling percentile
    volume_percentile = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).sum() / len(x[:-1]) if len(x[:-1]) > 0 else 0.5
    )
    
    # Volume Momentum Component
    volume_5d_ago = data['volume'].shift(5)
    volume_momentum = (data['volume'] - volume_5d_ago) / np.where(volume_5d_ago != 0, volume_5d_ago, 1)
    
    # 4. Liquidity Absorption Dynamics
    # Effective Price Calculation
    effective_price = data['amount'] / np.where(data['volume'] != 0, data['volume'], 1)
    
    # Absorption Pattern Detection
    effective_price_ma = effective_price.rolling(window=5, min_periods=1).mean()
    liquidity_absorption = (effective_price - effective_price_ma) / np.where(effective_price_ma != 0, effective_price_ma, 1)
    
    # 5. Previous Day Return Component
    prev_close = data['close'].shift(1)
    prev_prev_close = data['close'].shift(2)
    prev_day_return = (prev_close - prev_prev_close) / np.where(prev_prev_close != 0, prev_prev_close, 1)
    
    # 6. Factor Integration
    # Combine Volatility-Adjusted Reversal with Volume Signals
    morning_with_volume = vol_adjusted_morning * volume_percentile * (1 + volume_momentum)
    afternoon_with_volume = vol_adjusted_afternoon * volume_percentile * (1 + volume_momentum)
    
    # Incorporate Liquidity Absorption
    absorption_scaling = np.abs(liquidity_absorption)
    morning_with_liquidity = morning_with_volume * absorption_scaling
    afternoon_with_liquidity = afternoon_with_volume * absorption_scaling
    
    # Apply Previous Day Return Filter
    morning_final = morning_with_liquidity * prev_day_return
    afternoon_final = afternoon_with_liquidity * prev_day_return
    
    # Final Factor Construction
    combined_factor = (morning_final + afternoon_final) / 2
    bounded_factor = np.tanh(combined_factor)
    
    return bounded_factor
