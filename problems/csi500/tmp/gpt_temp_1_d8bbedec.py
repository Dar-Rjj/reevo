import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Range Efficiency
    daily_range = (data['high'] - data['low']) / data['close']
    range_autocorr = daily_range.rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 3 else np.nan, raw=False
    )
    volume_trend = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=False
    )
    factor1 = range_autocorr * volume_trend
    
    # Gap Decay Momentum
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_closure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    intraday_vol = (data['high'] - data['low']) / data['close']
    factor2 = opening_gap * gap_closure * intraday_vol
    
    # Volume-Weighted Price Acceleration
    returns = data['close'].pct_change()
    price_acceleration = returns.diff().rolling(window=3).mean()
    volume_momentum = data['volume'].pct_change(periods=3)
    factor3 = price_acceleration * volume_momentum
    
    # Intraday Reversal Confirmation
    intraday_return = (data['close'] - data['open']) / data['open']
    prev_day_move = data['close'].pct_change(1)
    volume_ratio = data['volume'] / data['volume'].rolling(window=5).mean()
    factor4 = intraday_return * np.sign(prev_day_move) * volume_ratio
    
    # Price-Volume Trend Divergence
    price_slope = data['close'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=False
    )
    volume_slope = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=False
    )
    factor5 = price_slope - volume_slope
    
    # CLV Persistence Signal
    clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']).replace(0, np.nan)
    clv_autocorr = clv.rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 3 else np.nan, raw=False
    )
    volume_trend_short = data['volume'].rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=False
    )
    factor6 = clv_autocorr * volume_trend_short
    
    # Dollar Volume Efficiency
    dollar_volume = data['close'] * data['volume']
    return_per_dollar = returns / dollar_volume.replace(0, np.nan)
    efficiency_momentum = return_per_dollar.rolling(window=5).mean()
    factor7 = efficiency_momentum
    
    # Multi-Timeframe Range Breakout
    short_term_range = (data['high'] - data['low']).rolling(window=3).mean()
    long_term_range = (data['high'] - data['low']).rolling(window=10).mean()
    range_ratio = short_term_range / long_term_range.replace(0, np.nan)
    volume_confirmation = data['volume'] / data['volume'].rolling(window=10).mean()
    factor8 = range_ratio * volume_confirmation
    
    # Combine factors with equal weights
    factors = pd.DataFrame({
        'f1': factor1, 'f2': factor2, 'f3': factor3, 'f4': factor4,
        'f5': factor5, 'f6': factor6, 'f7': factor7, 'f8': factor8
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=20, min_periods=10).mean()) / 
                                     x.rolling(window=20, min_periods=10).std())
    
    # Equal-weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
