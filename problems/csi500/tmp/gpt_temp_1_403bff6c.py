import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate previous day values
    data['Close_prev'] = data['close'].shift(1)
    data['High_prev'] = data['high'].shift(1)
    data['Low_prev'] = data['low'].shift(1)
    data['Open_prev'] = data['open'].shift(1)
    data['Volume_prev'] = data['volume'].shift(1)
    
    # Opening Pressure & Volatility Divergence
    data['Opening_Strength'] = (data['high'] - data['open']) / (data['open'] - data['low'])
    data['Opening_Strength'] = data['Opening_Strength'].replace([np.inf, -np.inf], np.nan)
    data['Volatility_Adjusted_Opening'] = data['Opening_Strength'] * (data['high'] - data['low']) / data['open']
    data['Pressure_Volatility_Divergence'] = data['Volatility_Adjusted_Opening'] * (data['Close_prev'] - data['open']) / (data['High_prev'] - data['Low_prev'])
    data['Pressure_Volatility_Divergence'] = data['Pressure_Volatility_Divergence'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Efficiency Volatility Integration
    data['Amount_per_Share'] = data['amount'] / data['volume']
    data['Amount_per_Share'] = data['Amount_per_Share'].replace([np.inf, -np.inf], np.nan)
    data['Volatility_Weighted_Efficiency'] = (data['close'] / data['Amount_per_Share']) * (data['high'] - data['low']) / data['open']
    data['Volatility_Weighted_Efficiency'] = data['Volatility_Weighted_Efficiency'].replace([np.inf, -np.inf], np.nan)
    data['Volume_Volatility_Intensity'] = data['Volatility_Weighted_Efficiency'] * data['volume'] / (data['high'] - data['low'])
    data['Volume_Volatility_Intensity'] = data['Volume_Volatility_Intensity'].replace([np.inf, -np.inf], np.nan)
    
    # Momentum-Volatility Compression Analysis
    data['Intraday_Compression'] = (data['high'] - data['low']) / (data['Close_prev'] - data['Open_prev'])
    data['Intraday_Compression'] = data['Intraday_Compression'].replace([np.inf, -np.inf], np.nan)
    data['Volatility_Elasticity'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) * (data['high'] - data['low']) / data['open']
    data['Volatility_Elasticity'] = data['Volatility_Elasticity'].replace([np.inf, -np.inf], np.nan)
    data['Volatility_Compression_Signal'] = data['Intraday_Compression'] * data['Volatility_Elasticity']
    
    # Volume-Pressure Volatility Dynamics
    data['Volume_Acceleration'] = data['volume'] / data['Volume_prev'] - 1
    data['Volume_Acceleration'] = data['Volume_Acceleration'].replace([np.inf, -np.inf], np.nan)
    data['Volatility_Distribution'] = (data['high'] - data['close']) * (data['high'] - data['low']) / data['open']
    data['Volume_Pressure_Volatility'] = data['Volume_Acceleration'] * data['Volatility_Distribution']
    
    # Volatility-Regime Pattern Detection
    data['Current_Volatility_Momentum'] = ((data['high'] - data['low']) / (data['high'] + data['low'])) * (data['high'] - data['low']) / data['open']
    data['Current_Volatility_Momentum'] = data['Current_Volatility_Momentum'].replace([np.inf, -np.inf], np.nan)
    data['Volatility_Acceleration_Difference'] = data['Current_Volatility_Momentum'] - data['Current_Volatility_Momentum'].shift(1)
    data['Volume_Volatility_Ratio'] = (data['volume'] / data['Volume_prev']) * (data['high'] - data['low']) / data['open']
    data['Volume_Volatility_Ratio'] = data['Volume_Volatility_Ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Time-Weighted Volatility Enhancement
    # Calculate Recent Volatility Activity Weight (5-day window)
    data['Daily_Volatility_Activity'] = (data['high'] - data['low']) * data['volume'] * (data['high'] - data['low']) / data['open']
    
    # Create exponential weights for 5-day window
    weights = np.exp(-np.arange(5) / 3)
    weights = weights / weights.sum()
    
    # Calculate weighted volatility activity
    data['Recent_Volatility_Activity_Weight'] = (
        data['Daily_Volatility_Activity'].rolling(window=5, min_periods=1).apply(
            lambda x: np.nansum(x * weights[:len(x)]), raw=False
        )
    )
    
    # Enhanced components
    data['Enhanced_Pressure_Volatility'] = data['Pressure_Volatility_Divergence'] * data['Recent_Volatility_Activity_Weight']
    data['Enhanced_Volatility_Compression'] = data['Volatility_Compression_Signal'] * data['Recent_Volatility_Activity_Weight']
    
    # Final Volatility-Convergence Factor Assembly
    # Core Volatility Signal Generation
    data['Combined_Volatility_Signal'] = data['Enhanced_Pressure_Volatility'] * data['Volume_Volatility_Intensity']
    data['Volatility_Compression_Adjustment'] = data['Combined_Volatility_Signal'] / data['Enhanced_Volatility_Compression']
    data['Volatility_Compression_Adjustment'] = data['Volatility_Compression_Adjustment'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Volatility Dynamics Integration
    data['Volume_Volatility_Persistence_Multiplier'] = (data['volume'] / data['Volume_prev']) * (data['high'] - data['low']) / data['open']
    data['Volume_Volatility_Persistence_Multiplier'] = data['Volume_Volatility_Persistence_Multiplier'].replace([np.inf, -np.inf], np.nan)
    data['Large_Trade_Volatility_Signal'] = (data['Amount_per_Share'] / data['close']) * (data['high'] - data['low']) / data['open']
    data['Large_Trade_Volatility_Signal'] = data['Large_Trade_Volatility_Signal'].replace([np.inf, -np.inf], np.nan)
    
    # Final Factor Construction
    data['Base_Volatility_Factor'] = data['Volatility_Compression_Adjustment'] * data['Volume_Volatility_Persistence_Multiplier']
    data['Final_Volatility_Convergence_Factor'] = (
        data['Base_Volatility_Factor'] * 
        data['Large_Trade_Volatility_Signal'] * 
        data['Volume_Pressure_Volatility'] * 
        data['Volatility_Acceleration_Difference']
    )
    
    # Return the final factor series
    return data['Final_Volatility_Convergence_Factor']
