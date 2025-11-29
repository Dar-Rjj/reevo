import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        current = data.loc[current_date]
        prev = data.loc[prev_date]
        
        # Intraday Momentum Divergence
        try:
            price_momentum = (current['close'] - current['open']) / (current['high'] - current['low'])
            # Assuming morning volume = first half of day, afternoon = second half
            # For simplicity, we'll use open-to-close vs close-to-close volume proxy
            daily_volume = current['volume']
            volume_divergence = (current['volume'] - prev['volume']) / daily_volume
            divergence_signal = price_momentum * volume_divergence
        except:
            divergence_signal = 0
        
        # Opening Range Breakout Efficiency
        try:
            breakout_efficiency = (current['high'] - current['open']) / (current['open'] - current['low'])
            # Using previous day's volume as proxy for opening concentration
            volume_concentration = current['volume'] / prev['volume'] if prev['volume'] > 0 else 1
            efficiency_score = breakout_efficiency * volume_concentration
        except:
            efficiency_score = 0
        
        # Volatility Compression Expansion
        try:
            current_range = current['high'] - current['low']
            prev_range = prev['high'] - prev['low']
            volatility_compression = (current_range - prev_range) / prev_range if prev_range > 0 else 0
            volume_burst = current['volume'] / prev['volume'] if prev['volume'] > 0 else 1
            compression_signal = volatility_compression * volume_burst
        except:
            compression_signal = 0
        
        # Price-Volume Fractal Dimension
        try:
            # Using daily range and volume as proxies
            price_fractal = (np.log(current['high']) - np.log(current['low'])) / np.log(current['volume'] + 1)
            volume_fractal = (np.log(current['volume']) - np.log(prev['volume'])) / np.log(2)  # 2 time periods
            fractal_signal = price_fractal * volume_fractal
        except:
            fractal_signal = 0
        
        # Liquidity Absorption
        try:
            # Using amount data for absorption
            absorption_ratio = (current['amount'] - prev['amount']) / current['amount'] if current['amount'] > 0 else 0
            # Volume skew proxy using price movement
            volume_skew = (current['close'] - current['open']) / current['volume'] if current['volume'] > 0 else 0
            absorption_signal = absorption_ratio * volume_skew
        except:
            absorption_signal = 0
        
        # Gap Persistence
        try:
            gap_persistence = (current['open'] - prev['close']) / (current['high'] - current['low'])
            volume_persistence = current['volume'] / prev['volume'] if prev['volume'] > 0 else 1
            persistence_strength = gap_persistence * volume_persistence
        except:
            persistence_strength = 0
        
        # Combine all signals with equal weights
        combined_signal = (
            divergence_signal + 
            efficiency_score + 
            compression_signal + 
            fractal_signal + 
            absorption_signal + 
            persistence_strength
        )
        
        result.loc[current_date] = combined_signal
    
    # Fill first value with 0
    if len(result) > 0:
        result.iloc[0] = 0
    
    return result
