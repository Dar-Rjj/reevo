import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Price Reversal Components
    data['intraday_reversal'] = (data['open'] - data['close']) / data['open']
    data['prev_close'] = data['close'].shift(1)
    data['gap_analysis'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Volatility Context - 10-day rolling standard deviation of Close returns
    data['volatility_context'] = data['returns'].rolling(window=10, min_periods=5).std()
    
    # Volume Dynamics
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_surge'] = data['volume'] / data['volume_median_20d']
    
    # Volatility-Weighted Volume
    data['daily_range'] = data['high'] - data['low']
    data['vol_weighted_volume'] = data['volume'] / data['daily_range']
    data['vol_weighted_volume'] = data['vol_weighted_volume'].replace([np.inf, -np.inf], np.nan)
    
    # Market State Analysis
    data['trend_direction'] = np.sign(data['returns'].rolling(window=20, min_periods=10).mean())
    data['daily_range_quantile'] = data['daily_range'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Rolling percentiles for volatility context
    data['volatility_percentile'] = data['volatility_context'].rolling(window=60, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Initialize factor column
    data['factor'] = np.nan
    
    # High Volatility Reversal Condition
    high_vol_condition = (
        (data['volatility_percentile'] > 0.6) &
        (data['gap_analysis'] < -0.02) &
        (data['volume_surge'] > 1.5)
    )
    data.loc[high_vol_condition, 'factor'] = (
        data.loc[high_vol_condition, 'intraday_reversal'] *
        data.loc[high_vol_condition, 'volume_surge'] *
        data.loc[high_vol_condition, 'volatility_context']
    )
    
    # Low Volatility Continuation Condition
    low_vol_condition = (
        (data['volatility_percentile'] < 0.4) &
        (data['gap_analysis'] > 0.01) &
        (data['vol_weighted_volume'] < 0.8)
    )
    data.loc[low_vol_condition, 'factor'] = (
        data.loc[low_vol_condition, 'gap_analysis'] *
        data.loc[low_vol_condition, 'vol_weighted_volume'] *
        data.loc[low_vol_condition, 'trend_direction']
    )
    
    # Default condition for other cases
    other_condition = data['factor'].isna()
    data.loc[other_condition, 'factor'] = (
        data.loc[other_condition, 'intraday_reversal'] *
        data.loc[other_condition, 'vol_weighted_volume']
    )
    
    # Return factor series
    return data['factor']
