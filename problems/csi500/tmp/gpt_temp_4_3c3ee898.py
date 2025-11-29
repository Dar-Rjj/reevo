import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic intraday components
    data['upside_range'] = data['high'] - data['open']
    data['downside_range'] = data['open'] - data['low']
    data['daily_range'] = data['high'] - data['low']
    
    # Price Movement Skewness
    data['price_skew'] = np.where(data['daily_range'] > 0,
                                 (data['upside_range'] - data['downside_range']) / data['daily_range'],
                                 0)
    
    # Skew persistence (3-day rolling mean)
    data['skew_persistence'] = data['price_skew'].rolling(window=3, min_periods=1).mean()
    
    # Mid price calculation (assuming mid-point of trading day)
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Price acceleration rates
    data['morning_acceleration'] = np.where(data['open'] > 0,
                                          (data['mid_price'] - data['open']) / data['open'],
                                          0)
    data['afternoon_acceleration'] = np.where(data['mid_price'] > 0,
                                            (data['close'] - data['mid_price']) / data['mid_price'],
                                            0)
    
    # Acceleration divergence
    data['acceleration_divergence'] = data['morning_acceleration'] - data['afternoon_acceleration']
    
    # Volume concentration (assuming equal split for morning/afternoon)
    # Since we don't have intraday volume, we'll use proxy measures
    data['volume_imbalance'] = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if x[:-1].std() > 0 else 0
    )
    
    # Range compression patterns
    data['prev_range'] = data['daily_range'].shift(1)
    data['range_ratio'] = np.where(data['prev_range'] > 0,
                                 data['daily_range'] / data['prev_range'],
                                 1)
    
    # Range compression persistence
    data['range_compression'] = data['range_ratio'].rolling(window=3, min_periods=1).apply(
        lambda x: (x < 1).sum() / len(x)
    )
    
    # Price position efficiency
    data['closing_efficiency'] = np.where(data['daily_range'] > 0,
                                        (data['close'] - data['low']) / data['daily_range'],
                                        0.5)
    data['opening_efficiency'] = np.where(data['daily_range'] > 0,
                                        (data['open'] - data['low']) / data['daily_range'],
                                        0.5)
    
    # Range utilization (how much of the range was actually used)
    data['range_utilization'] = (abs(data['close'] - data['open']) + 
                               abs(data['high'] - data['low'])) / (2 * data['daily_range'])
    data['range_utilization'] = np.where(data['daily_range'] > 0, data['range_utilization'], 0.5)
    
    # Volume acceleration proxy (using rolling volume changes)
    data['volume_momentum'] = data['volume'].pct_change(periods=1).rolling(window=3, min_periods=1).mean()
    
    # Construct composite alpha factors
    
    # Primary asymmetry signal
    data['price_volume_asymmetry'] = (
        data['price_skew'] * data['volume_imbalance'] * 
        (data['morning_acceleration'] - data['afternoon_acceleration'])
    )
    
    # Range dynamics enhancement
    data['range_context_signal'] = (
        data['range_compression'] * data['price_skew'] * 
        data['acceleration_divergence']
    )
    
    # Multi-timeframe composite factor
    data['short_term_momentum'] = (
        data['price_skew'].rolling(window=3, min_periods=1).mean() *
        data['volume_momentum'].rolling(window=3, min_periods=1).mean()
    )
    
    data['medium_term_range'] = (
        data['range_ratio'].rolling(window=5, min_periods=1).mean() *
        data['range_utilization'].rolling(window=5, min_periods=1).mean()
    )
    
    # Final alpha factor combining all components
    alpha_factor = (
        0.4 * data['price_volume_asymmetry'] +
        0.3 * data['range_context_signal'] +
        0.2 * data['short_term_momentum'] +
        0.1 * data['medium_term_range']
    )
    
    # Clean up any infinite or NaN values
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return alpha_factor
