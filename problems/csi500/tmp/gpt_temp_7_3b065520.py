import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Morning Momentum Component
    data['Morning_High'] = data['high'].rolling(window=120, min_periods=1).apply(lambda x: x[:60].max() if len(x) >= 60 else np.nan)
    data['Morning_Low'] = data['low'].rolling(window=120, min_periods=1).apply(lambda x: x[:60].min() if len(x) >= 60 else np.nan)
    data['Morning_Range'] = (data['Morning_High'] - data['Morning_Low']) / data['open']
    data['Morning_Midpoint'] = (data['Morning_High'] + data['Morning_Low']) / 2
    data['Morning_Return'] = (data['Morning_Midpoint'] - data['open']) / data['open']
    
    # Acceleration Component
    data['Midday_Price'] = (data['high'] + data['low']) / 2
    denominator = np.abs(data['Midday_Price'] - data['open'])
    denominator = np.where(denominator == 0, 1e-8, denominator)  # Avoid division by zero
    data['Price_Acceleration'] = (data['close'] - data['Midday_Price']) / denominator
    
    # Reversal Integration
    data['Previous_Day_Return'] = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    data['Acceleration_Reversal_Signal'] = data['Price_Acceleration'] * data['Previous_Day_Return'] * data['Morning_Return']
    
    # Volatility Adjustment System
    data['Current_Intraday_Volatility'] = (data['high'] - data['low']) / data['close']
    data['Historical_Volatility_Baseline'] = data['Current_Intraday_Volatility'].rolling(window=20, min_periods=1).mean()
    data['Volatility_Adjusted_Signal'] = data['Acceleration_Reversal_Signal'] * (data['Current_Intraday_Volatility'] / data['Historical_Volatility_Baseline'])
    
    # Volume Confirmation Framework
    data['Volume_to_Volatility_Ratio'] = data['volume'] / data['Current_Intraday_Volatility']
    data['Volume_Deviation'] = data['Volume_to_Volatility_Ratio'] / data['Volume_to_Volatility_Ratio'].rolling(window=5, min_periods=1).mean()
    
    data['Volume_Ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['Volume_Acceleration'] = data['Volume_Ratio'] - data['Volume_Ratio'].shift(1)
    data['Volume_Confirmation_Signal'] = data['Volume_Deviation'] * data['Volume_Acceleration']
    
    # Price Extremes Filter
    data['High_Distance'] = (data['high'] - data['close']) / data['close']
    data['Low_Distance'] = (data['close'] - data['low']) / data['close']
    data['Max_Distance'] = np.maximum(data['High_Distance'], data['Low_Distance'])
    data['Direction_Sign'] = np.where(data['High_Distance'] > data['Low_Distance'], -1, 1)
    
    # Composite Factor Construction
    data['Core_Signal'] = data['Volatility_Adjusted_Signal'] * data['Volume_Confirmation_Signal']
    data['Extremes_Adjustment'] = data['Direction_Sign'] * data['Max_Distance']
    data['Final_Factor'] = data['Core_Signal'] * data['Extremes_Adjustment'] * data['Morning_Range']
    
    return data['Final_Factor']
