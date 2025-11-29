import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Dynamics Analysis
    data['prev_close'] = data['close'].shift(1)
    data['raw_gap'] = data['open'] - data['prev_close']
    data['gap_abs'] = data['raw_gap'].abs()
    
    # Recent volatility for gap quality assessment
    data['volatility_5d'] = data['close'].pct_change().rolling(window=5).std()
    data['gap_quality'] = data['gap_abs'] / (data['volatility_5d'] * data['prev_close'] + 1e-8)
    
    # Gap persistence tracking
    data['gap_persistence'] = data['raw_gap'].rolling(window=3).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 3 and not np.isnan(x).any() else 0
    )
    
    # Range Efficiency Measurement
    data['daily_range'] = data['high'] - data['low']
    data['range_utilization'] = (data['close'] - data['open']).abs() / (data['daily_range'] + 1e-8)
    
    # Range pattern analysis
    data['range_trend'] = data['daily_range'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
    )
    
    # Volume Confirmation
    # Morning volume concentration (first hour proxy - using opening data)
    data['volume_open_ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Volume-gap direction alignment
    data['volume_gap_alignment'] = np.sign(data['raw_gap']) * data['volume_open_ratio']
    
    # Amount-based signals
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['large_trade_concentration'] = data['avg_trade_size'] / data['avg_trade_size'].rolling(window=10).mean()
    
    # Institutional flow patterns (amount volatility)
    data['amount_volatility'] = data['amount'].pct_change().rolling(window=5).std()
    
    # Price Acceleration Integration
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['momentum_acceleration'] = data['intraday_return'].diff().rolling(window=3).mean()
    
    # Gap-driven acceleration patterns
    data['gap_acceleration'] = data['raw_gap'] * data['momentum_acceleration']
    
    # Acceleration quality assessment
    data['acceleration_quality'] = data['momentum_acceleration'].abs() / (data['volatility_5d'] + 1e-8)
    
    # Composite Factor Generation
    # Gap-Range Efficiency component
    gap_range_efficiency = (
        data['gap_quality'] * 
        data['range_utilization'] * 
        np.sign(data['raw_gap'])
    )
    
    # Volume Confirmation component
    volume_confirmation = (
        data['volume_gap_alignment'] * 
        data['large_trade_concentration'] * 
        (1 + data['amount_volatility'])
    )
    
    # Acceleration component
    acceleration_component = (
        data['gap_acceleration'] * 
        data['acceleration_quality']
    )
    
    # Multi-dimensional combination
    composite_factor = (
        gap_range_efficiency * volume_confirmation + 
        acceleration_component
    )
    
    # Handle any remaining NaN values
    composite_factor = composite_factor.fillna(0)
    
    return composite_factor
