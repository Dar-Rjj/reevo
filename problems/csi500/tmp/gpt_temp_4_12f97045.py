import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Volatility Compression Breakout Signal
    # Calculate daily range ratio
    data['range_ratio'] = (data['high'] - data['low']) / data['close']
    
    # Calculate rolling average range ratio (5-day)
    data['avg_range_ratio_5'] = data['range_ratio'].rolling(window=5, min_periods=1).mean()
    
    # Calculate compression score
    data['compression_score'] = data['range_ratio'] / data['avg_range_ratio_5']
    
    # Calculate breakout magnitude
    data['breakout_magnitude'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['breakout_magnitude'] = data['breakout_magnitude'].replace([np.inf, -np.inf], np.nan)
    
    # Final volatility compression breakout signal
    data['vol_breakout_signal'] = data['breakout_magnitude'] * (1 / data['compression_score'])
    
    # Price-Volume Divergence Momentum
    # Calculate price momentum
    data['price_momentum'] = (data['close'] - data['open']) / data['close']
    
    # Calculate volume momentum
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_momentum'] = data['volume'] / data['prev_volume']
    data['volume_momentum'] = data['volume_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate rolling correlation between price and volume momentum (5-day)
    corr_values = []
    for i in range(len(data)):
        if i < 4:
            corr_values.append(0)
        else:
            window_data = data.iloc[i-4:i+1]
            price_window = window_data['price_momentum']
            volume_window = window_data['volume_momentum']
            valid_mask = (~price_window.isna()) & (~volume_window.isna())
            if valid_mask.sum() >= 3:
                corr = price_window[valid_mask].corr(volume_window[valid_mask])
                corr_values.append(corr if not np.isnan(corr) else 0)
            else:
                corr_values.append(0)
    
    data['price_volume_corr'] = corr_values
    
    # Calculate divergence strength
    data['divergence_strength'] = abs(data['price_momentum']) * (1 - data['price_volume_corr'])
    
    # Final price-volume divergence factor
    data['price_volume_divergence'] = data['price_momentum'] * data['divergence_strength']
    
    # Intraday Reversal Strength
    # Calculate early strength
    data['early_strength'] = (data['high'] - data['open']) / (data['high'] - data['low'])
    data['early_strength'] = data['early_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate late weakness
    data['late_weakness'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['late_weakness'] = data['late_weakness'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate volume concentration
    data['volume_concentration'] = data['volume'] / (data['high'] - data['low'])
    data['volume_concentration'] = data['volume_concentration'].replace([np.inf, -np.inf], np.nan)
    
    # Final intraday reversal factor
    data['intraday_reversal'] = (data['early_strength'] - data['late_weakness']) * data['volume_concentration']
    
    # Multi-timeframe Efficiency Ratio
    # Calculate short-term efficiency
    data['short_term_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['short_term_efficiency'] = data['short_term_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate medium-term efficiency
    data['close_5'] = data['close'].shift(5)
    data['max_high_5'] = data['high'].rolling(window=5, min_periods=1).max()
    data['min_low_5'] = data['low'].rolling(window=5, min_periods=1).min()
    data['medium_term_efficiency'] = abs(data['close'] - data['close_5']) / (data['max_high_5'] - data['min_low_5'])
    data['medium_term_efficiency'] = data['medium_term_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate efficiency divergence
    data['efficiency_divergence'] = data['short_term_efficiency'] - data['medium_term_efficiency']
    
    # Final multi-timeframe efficiency factor
    data['efficiency_ratio'] = data['efficiency_divergence'] * (data['short_term_efficiency'] / data['medium_term_efficiency'])
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Combine all factors with equal weighting
    factors = ['vol_breakout_signal', 'price_volume_divergence', 'intraday_reversal', 'efficiency_ratio']
    
    for factor in factors:
        data[factor] = data[factor].fillna(0)
    
    # Final combined factor (simple average of normalized factors)
    result = (data['vol_breakout_signal'] + data['price_volume_divergence'] + 
              data['intraday_reversal'] + data['efficiency_ratio']) / 4
    
    return result
