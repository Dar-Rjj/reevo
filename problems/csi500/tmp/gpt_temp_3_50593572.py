import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Breakout Detection Component
    data['Close_prev'] = data['close'].shift(1)
    data['True_Range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['Close_prev']),
            np.abs(data['low'] - data['Close_prev'])
        )
    )
    
    data['Volatility_Breakout_Signal'] = (data['close'] - data['open']) / data['True_Range']
    data['Volatility_Expansion'] = data['True_Range'] / data['True_Range'].shift(1)
    data['Volatility_Regime_Shift'] = data['Volatility_Expansion'] * data['Volatility_Breakout_Signal']
    
    # Price Path Efficiency Component
    data['Open_to_Close_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['High_Low_Capture_Ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['Path_Deviation'] = np.abs(data['Open_to_Close_Efficiency'] - data['High_Low_Capture_Ratio'])
    data['Efficiency_Momentum'] = data['Path_Deviation'] - data['Path_Deviation'].shift(1)
    
    # Volume Flow Regime Confirmation
    data['Volume_Intensity'] = data['volume'] / data['volume'].shift(1)
    data['Volume_Price_Alignment'] = np.sign(data['close'] - data['open']) * data['Volume_Intensity']
    
    data['Trade_Size_Indicator'] = data['amount'] / data['volume']
    data['Institutional_Flow'] = data['Trade_Size_Indicator'] * data['Volume_Intensity']
    
    # Volume Regime Multiplier
    data['Volume_Regime_Multiplier'] = data['Volume_Price_Alignment'] * data['Institutional_Flow']
    
    # Multi-Timeframe Regime Momentum
    data['Short_Term_Regime_Momentum'] = data['Volatility_Regime_Shift'] - data['Volatility_Regime_Shift'].shift(2)
    data['Regime_Persistence'] = data['Volatility_Regime_Shift'].rolling(window=3, min_periods=1).std()
    
    data['Medium_Term_Regime_Strength'] = data['Volatility_Regime_Shift'].rolling(window=5, min_periods=1).mean()
    data['Regime_Consistency'] = data['Volatility_Regime_Shift'].rolling(window=5, min_periods=1).std()
    
    data['Cross_Timeframe_Alignment'] = (
        data['Short_Term_Regime_Momentum'] * data['Medium_Term_Regime_Strength']
    ) / (data['Regime_Consistency'] + 1e-8)
    
    # Volatility-Efficiency Convergence Factor
    data['Volatility_Efficiency_Convergence'] = (
        data['Volatility_Regime_Shift'] * data['Efficiency_Momentum']
    ) / (data['Path_Deviation'] + 1e-8)
    
    # Cross-sectional ranking calculations
    def cross_sectional_rank(group):
        regime_strength = group['Volatility_Regime_Shift'].rank(pct=True)
        efficiency_rank = group['Efficiency_Momentum'].rank(pct=True)
        volume_rank = group['Volume_Regime_Multiplier'].rank(pct=True)
        
        regime_leadership = (regime_strength * efficiency_rank * volume_rank)
        cross_sectional_strength = regime_leadership.rank(pct=True)
        
        return cross_sectional_strength
    
    # Apply cross-sectional ranking by date
    cross_sectional_results = []
    for date, group in data.groupby(data.index):
        ranked_group = cross_sectional_rank(group)
        ranked_group.name = date
        cross_sectional_results.append(ranked_group)
    
    cross_sectional_ranks = pd.concat(cross_sectional_results)
    data['Cross_Sectional_Strength'] = cross_sectional_ranks
    
    # Final Alpha Factor Generation
    data['Regime_Momentum_Factor'] = (
        data['Volatility_Efficiency_Convergence'] * 
        data['Cross_Sectional_Strength'] * 
        data['Volume_Regime_Multiplier'] * 
        data['Cross_Timeframe_Alignment']
    )
    
    # Dynamic weighting based on regime characteristics
    volatility_weight = np.abs(data['Volatility_Regime_Shift']) / (
        np.abs(data['Volatility_Regime_Shift']).rolling(window=10, min_periods=1).mean() + 1e-8
    )
    
    volume_weight = np.abs(data['Volume_Regime_Multiplier']) / (
        np.abs(data['Volume_Regime_Multiplier']).rolling(window=10, min_periods=1).mean() + 1e-8
    )
    
    # Final factor with dynamic weighting
    data['Final_Factor'] = (
        data['Regime_Momentum_Factor'] * 
        np.clip(volatility_weight, 0.5, 2.0) * 
        np.clip(volume_weight, 0.5, 2.0)
    )
    
    # Clean up intermediate columns and return final factor
    result = data['Final_Factor'].copy()
    
    return result
