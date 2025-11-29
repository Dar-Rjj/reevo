import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Calculate Intraday Price Strength
    # Daily price efficiency: (Close - Low) / (High - Low)
    price_efficiency = (data['close'] - data['low']) / (data['high'] - data['low'])
    # Replace infinite values with NaN
    price_efficiency = price_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # 3-day average of price efficiency
    intraday_strength = price_efficiency.rolling(window=3, min_periods=1).mean()
    
    # Analyze Volume-Price Divergence
    def calculate_trend(series, window=5):
        """Calculate linear regression slope for trend"""
        trends = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= window - 1:
                y_values = series.iloc[i-window+1:i+1].values
                if len(y_values) == window and not np.any(np.isnan(y_values)):
                    x_values = np.arange(window)
                    slope, _, _, _, _ = linregress(x_values, y_values)
                    trends.iloc[i] = slope
                else:
                    trends.iloc[i] = np.nan
            else:
                trends.iloc[i] = np.nan
        return trends
    
    # Compute 5-day volume trend
    volume_trend = calculate_trend(data['volume'], window=5)
    
    # Compute 5-day price trend
    price_trend = calculate_trend(data['close'], window=5)
    
    # Calculate divergence as volume trend minus price trend
    divergence = volume_trend - price_trend
    
    # Combine Intraday Strength and Divergence Signals
    # Multiply intraday strength by divergence
    combined_signal = intraday_strength * divergence
    
    # Apply hyperbolic tangent for bounded output
    bounded_signal = np.tanh(combined_signal)
    
    # Weight by recent trading activity using amount (3-day average)
    amount_weight = data['amount'].rolling(window=3, min_periods=1).mean()
    
    # Final factor: bounded signal weighted by trading activity
    factor = bounded_signal * amount_weight
    
    return factor
