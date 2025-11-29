import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate all required components
    # Intraday Range Efficiency Momentum
    intraday_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Volume-weighted persistence
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_ratio = data['volume'] / volume_ma_20
    
    # Count consecutive same-direction days
    direction = np.sign(data['close'] - data['open'])
    consecutive_days = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if direction.iloc[i] == direction.iloc[i-1]:
            consecutive_days.iloc[i] = consecutive_days.iloc[i-1] + 1
    
    intraday_momentum = intraday_efficiency * consecutive_days * volume_ratio
    
    # Gap-Volatility Adjusted Momentum
    gap_pct = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Average True Range (10-day)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = true_range.rolling(window=10, min_periods=1).mean()
    
    # 3-day Price Rate of Change
    roc_3 = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    gap_momentum = gap_pct * atr_10 * roc_3
    
    # Volume-Amplitude Correlation Factor
    price_amplitude = (data['high'] - data['low']) / data['open']
    price_amplitude = price_amplitude.replace([np.inf, -np.inf], np.nan)
    
    volume_median_10 = data['volume'].rolling(window=10, min_periods=1).median()
    volume_surprise = data['volume'] / volume_median_10
    
    # 5-day correlation between amplitude and volume surprise
    corr_window = 5
    amplitude_volume_corr = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= corr_window - 1:
            start_idx = i - corr_window + 1
            window_data = data.iloc[start_idx:i+1]
            corr_val = window_data.assign(
                amplitude=price_amplitude.iloc[start_idx:i+1],
                surprise=volume_surprise.iloc[start_idx:i+1]
            )[['amplitude', 'surprise']].corr().iloc[0,1]
            amplitude_volume_corr.iloc[i] = corr_val
    
    volume_amplitude_factor = price_amplitude * volume_surprise * amplitude_volume_corr
    
    # Accumulation-Distribution Efficiency
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    
    # 3-day Money Flow Change
    money_flow_change = money_flow - money_flow.shift(3)
    
    # Weight by Amount / (High - Low)
    amount_weight = data['amount'] / (data['high'] - data['low'])
    amount_weight = amount_weight.replace([np.inf, -np.inf], np.nan)
    
    accumulation_factor = money_flow_change * amount_weight
    
    # Breakout-Volume Confirmation Factor
    prev_high = data['high'].shift(1)
    prev_low = data['low'].shift(1)
    
    # Identify breakouts
    high_breakout = data['close'] > prev_high
    low_breakout = data['close'] < prev_low
    breakout_signal = high_breakout.astype(int) - low_breakout.astype(int)
    
    # Count consecutive breakout days
    consecutive_breakouts = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if breakout_signal.iloc[i] != 0 and breakout_signal.iloc[i] == breakout_signal.iloc[i-1]:
            consecutive_breakouts.iloc[i] = consecutive_breakouts.iloc[i-1] + 1
    
    breakout_factor = breakout_signal * volume_ratio * consecutive_breakouts
    
    # Price-Volume Divergence Momentum
    price_roc_3 = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    volume_roc_3 = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    
    divergence = price_roc_3 - volume_roc_3
    
    # 10-day Average Daily Range
    daily_range = (data['high'] - data['low']) / data['open']
    avg_daily_range_10 = daily_range.rolling(window=10, min_periods=1).mean()
    
    divergence_momentum = divergence * avg_daily_range_10
    
    # Liquidity-Weighted Return Acceleration
    return_3 = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    return_10 = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    return_acceleration = return_3 - return_10
    
    # Liquidity weighting
    liquidity = data['amount'] / (data['high'] - data['low'])
    liquidity = liquidity.replace([np.inf, -np.inf], np.nan)
    
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ma_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_ratio_5_20 = volume_ma_5 / volume_ma_20
    
    liquidity_factor = return_acceleration * liquidity * volume_ratio_5_20
    
    # Efficiency-Persistence Composite
    open_close_efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    open_close_efficiency = open_close_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Count consecutive high-efficiency days (efficiency > 0.7)
    high_efficiency = open_close_efficiency > 0.7
    consecutive_high_eff = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if high_efficiency.iloc[i] and high_efficiency.iloc[i-1]:
            consecutive_high_eff.iloc[i] = consecutive_high_eff.iloc[i-1] + 1
    
    volume_median_10 = data['volume'].rolling(window=10, min_periods=1).median()
    volume_ratio_median = data['volume'] / volume_median_10
    
    efficiency_composite = open_close_efficiency * consecutive_high_eff * volume_ratio_median
    
    # Combine all factors with equal weights
    factors = [
        intraday_momentum,
        gap_momentum,
        volume_amplitude_factor,
        accumulation_factor,
        breakout_factor,
        divergence_momentum,
        liquidity_factor,
        efficiency_composite
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=data.index, dtype=float)
    for f in factors:
        f_normalized = (f - f.mean()) / f.std()
        combined_factor += f_normalized
    
    # Final normalization
    factor = (combined_factor - combined_factor.mean()) / combined_factor.std()
    
    return factor
