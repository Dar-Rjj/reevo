import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum-Rejection Divergence Component
    data['AM_Momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['PM_Rejection_Signal'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['Lower_Shadow_Rejection'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Calculate rolling correlation for momentum-rejection divergence
    am_momentum_series = data['AM_Momentum']
    pm_rejection_series = data['PM_Rejection_Signal']
    
    # Manual rolling correlation calculation to avoid future data
    momentum_rejection_corr = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:  # Need at least 3 points for correlation
            window_am = am_momentum_series.iloc[i-2:i+1]
            window_pm = pm_rejection_series.iloc[i-2:i+1]
            if len(window_am) == 3 and len(window_pm) == 3:
                corr_val = window_am.corr(window_pm)
                momentum_rejection_corr.iloc[i] = corr_val if not np.isnan(corr_val) else 0
            else:
                momentum_rejection_corr.iloc[i] = 0
        else:
            momentum_rejection_corr.iloc[i] = 0
    
    data['Momentum_Rejection_Divergence'] = (
        abs(data['AM_Momentum'] - data['PM_Rejection_Signal']) * momentum_rejection_corr
    )
    
    # Volatility-Weighted Volume Acceleration
    data['Volume_Flow_Direction'] = np.sign((data['close'] - data['open']) * data['volume'])
    
    # Calculate rolling mean of volume
    data['Volume_MA_3'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['Volume_Velocity'] = data['volume'] / data['Volume_MA_3']
    data['Volume_Acceleration'] = data['Volume_Velocity'] * data['Volume_Flow_Direction']
    
    # Volatility Context
    data['Current_Day_Volatility'] = (data['high'] - data['low']) / data['open']
    data['High_Low_Range_MA_5'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['Volatility_Expansion'] = (data['high'] - data['low']) / data['High_Low_Range_MA_5']
    
    data['Volatility_Weighted_Volume'] = (
        data['Volume_Acceleration'] * data['Current_Day_Volatility'] * data['Volume_Flow_Direction']
    )
    
    # Price-Volume Compression Analysis
    data['High_Low_1d_ago'] = (data['high'] - data['low']).shift(1)
    data['Volume_1d_ago'] = data['volume'].shift(1)
    
    # Handle division by zero
    data['Range_Compression_Ratio'] = np.where(
        data['High_Low_1d_ago'] > 0,
        (data['high'] - data['low']) / data['High_Low_1d_ago'],
        1.0
    )
    data['Volume_Compression'] = np.where(
        data['Volume_1d_ago'] > 0,
        data['volume'] / data['Volume_1d_ago'],
        1.0
    )
    
    data['Compression_Intensity'] = data['Range_Compression_Ratio'] * data['Volume_Compression']
    data['Price_Volume_Efficiency'] = (
        (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'])
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Multi-Timeframe Confirmation Framework
    data['Close_5d_ago'] = data['close'].shift(5)
    data['Price_Momentum_Consistency'] = (
        (data['close'] - data['open']) - (data['close'] - data['Close_5d_ago']) / 5
    )
    data['Direction_Alignment'] = (
        np.sign(data['close'] - data['open']) * np.sign(data['close'] - data['Close_5d_ago'])
    )
    
    # Volume Pattern Alignment
    data['Volume_MA_3_alt'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['Volume_5d_ago'] = data['volume'].shift(5)
    data['Short_vs_Medium_Volume'] = (
        data['volume'] / data['Volume_MA_3_alt'] - 
        np.where(data['Volume_5d_ago'] > 0, data['volume'] / data['Volume_5d_ago'], 0)
    )
    
    data['Volume_Acceleration_MA_3'] = data['Volume_Acceleration'].rolling(window=3, min_periods=1).mean()
    data['Volume_Acceleration_Consistency'] = (
        data['Volume_Acceleration'] * data['Volume_Acceleration_MA_3']
    )
    
    # Regime-Adaptive Signal Weighting
    data['Regime_Weight'] = 1.0
    data.loc[data['Volatility_Expansion'] > 1.2, 'Regime_Weight'] = 1.5
    data.loc[data['Volatility_Expansion'] < 0.8, 'Regime_Weight'] = 0.7
    
    # Composite Factor Synthesis
    # Core Divergence-Volume Score
    data['Core_Divergence_Volume_Score'] = (
        data['Momentum_Rejection_Divergence'] * data['Volatility_Weighted_Volume'] +
        data['Lower_Shadow_Rejection'] * data['Volume_Flow_Direction']
    )
    
    # Compression-Adjusted Signal
    data['Compression_Adjusted_Signal'] = (
        data['Core_Divergence_Volume_Score'] * data['Compression_Intensity'] +
        data['Price_Volume_Efficiency'] * data['Volume_Acceleration']
    )
    
    # Multi-Timeframe Confirmation
    data['Confirmed_Signal'] = (
        data['Compression_Adjusted_Signal'] * data['Price_Momentum_Consistency'] +
        data['Short_vs_Medium_Volume'] * data['Volume_Acceleration_Consistency']
    )
    
    # Regime-Adaptive Final Score
    data['Regime_Adaptive_Final_Score'] = (
        data['Confirmed_Signal'] * data['Regime_Weight'] * 
        np.where(data['PM_Rejection_Signal'] > 0.5, -1, 1)  # PM Rejection Direction Adjustment
    )
    
    # Final Factor Output
    factor = data['Regime_Adaptive_Final_Score']
    
    return factor
