import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Breakout-Reversal Analysis
    # Compute Breakout Components
    df['High_Breakout'] = df['high'] / df['high'].rolling(window=5).max()
    df['Low_Breakout'] = df['low'] / df['low'].rolling(window=5).min()
    
    # Calculate Reversal Signals
    df['Intraday_Reversal'] = (df['high'] + df['low']) / 2 - df['close']
    df['Volume_Weighted_Reversal'] = df['Intraday_Reversal'] * df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Derive Momentum Divergence
    df['AM_Momentum'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['PM_Momentum'] = (df['close'] - (df['high'] + df['low'])/2) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate 3-day rolling correlation between AM and PM Momentum
    rolling_corr = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if i >= 2:
            window_data = df.iloc[max(0, i-2):i+1]
            if len(window_data) >= 3:
                corr_val = window_data['AM_Momentum'].corr(window_data['PM_Momentum'])
                rolling_corr.iloc[i] = corr_val if not np.isnan(corr_val) else 0
            else:
                rolling_corr.iloc[i] = 0
        else:
            rolling_corr.iloc[i] = 0
    
    df['Momentum_Divergence'] = rolling_corr * abs(df['AM_Momentum'] - df['PM_Momentum'])
    
    # Efficiency and Volume System
    # Price Efficiency Components
    df['Price_Range_Efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['Volume_Efficiency'] = df['amount'] / (df['volume'] * df['high']).replace(0, np.nan)
    
    # Volume Acceleration Signals
    df['Volume_Velocity'] = np.sign((df['close'] - df['open']) * df['volume']) * (df['volume'] / df['volume'].rolling(window=3).mean())
    df['Volume_Momentum'] = ((df['close'] - df['open']) * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # Derive Combined Efficiency
    df['Efficiency_Score'] = df['Price_Range_Efficiency'] * df['Volume_Efficiency']
    df['Volume_Acceleration'] = df['Volume_Velocity'] * df['Volume_Momentum']
    
    # Volatility and Divergence Framework
    # Volatility Regime Analysis
    df['Short_Term_Volatility'] = df['close'].rolling(window=3).std()
    df['Long_Term_Volatility'] = df['close'].rolling(window=20).std()
    
    # Price-Volume Divergence
    df['5_day_Price_Change'] = df['close'] / df['close'].shift(5) - 1
    df['5_day_Volume_Change'] = df['volume'] / df['volume'].shift(5) - 1
    df['Price_Volume_Divergence'] = df['5_day_Price_Change'] - df['5_day_Volume_Change']
    
    # Derive Market Regime Signals
    df['Volatility_Regime'] = df['Short_Term_Volatility'] / df['Long_Term_Volatility']
    df['Divergence_Signal'] = df['Price_Volume_Divergence'] * df['Volume_Efficiency']
    
    # Composite Factor Generation
    df['Breakout_Reversal_Component'] = (df['High_Breakout'] - df['Low_Breakout']) * df['Momentum_Divergence'] * df['Volume_Weighted_Reversal']
    df['Efficiency_Component'] = df['Efficiency_Score'] * df['Volume_Acceleration']
    df['Regime_Component'] = df['Volatility_Regime'] * df['Divergence_Signal']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['Breakout_Reversal_Component'] * df['Efficiency_Component'] * df['Regime_Component']
    
    return df['alpha_factor']
