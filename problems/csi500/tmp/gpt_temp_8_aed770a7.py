import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price components
    df['PrevClose'] = df.groupby(level=1)['close'].shift(1)
    df['PrevOpen'] = df.groupby(level=1)['open'].shift(1)
    
    # Calculate True Range
    df['TR1'] = df['high'] - df['low']
    df['TR2'] = abs(df['high'] - df['PrevClose'])
    df['TR3'] = abs(df['low'] - df['PrevClose'])
    df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['PrevTrueRange'] = df.groupby(level=1)['TrueRange'].shift(1)
    
    # Assume midday price as average of open and high for AM session
    df['Midday'] = (df['open'] + df['high']) / 2
    
    # Calculate AM and PM ranges
    df['AM_Range'] = df['high'] - df['open']
    df['PM_Range'] = df['close'] - df['low']
    
    # Gap momentum fragmentation
    df['Gap_Momentum'] = (df['open'] / df['PrevClose'] - 1)
    df['AM_PM_Diff'] = abs((df['Midday'] - df['open']) - (df['close'] - df['Midday']))
    df['Gap_Momentum_Fragmentation'] = df['Gap_Momentum'] * df['AM_PM_Diff']
    
    # Range utilization divergence
    df['Range_Utilization'] = abs(df['close'] - df['open']) / df['TrueRange']
    df['Prev_Range_Utilization'] = df.groupby(level=1)['Range_Utilization'].shift(1)
    df['Range_Utilization_Divergence'] = df['Range_Utilization'] - df['Prev_Range_Utilization']
    
    # Gap-fragmentation momentum
    df['Gap_Fragmentation_Momentum'] = df['Gap_Momentum_Fragmentation'] * df['Range_Utilization_Divergence']
    
    # AM-PM fragmentation ratio
    df['AM_Fragmentation'] = abs(df['Midday'] - df['open'])
    df['PM_Fragmentation'] = abs(df['close'] - df['Midday'])
    df['AM_PM_Fragmentation_Ratio'] = df['AM_Fragmentation'] / (df['PM_Fragmentation'] + 1e-8)
    
    # Range efficiency fragmentation
    df['AM_Range_Utilization'] = df['AM_Fragmentation'] / (df['AM_Range'] + 1e-8)
    df['PM_Range_Utilization'] = df['PM_Fragmentation'] / (df['PM_Range'] + 1e-8)
    df['Range_Efficiency_Fragmentation'] = df['AM_Range_Utilization'] - df['PM_Range_Utilization']
    
    # Gap-fragmentation persistence (3-day trend)
    df['Gap_Fragmentation_Momentum_3d'] = df.groupby(level=1)['Gap_Fragmentation_Momentum'].rolling(window=3, min_periods=1).mean().reset_index(level=1, drop=True)
    
    # Fragmentation alignment
    df['AM_Fragmentation_Sign'] = np.sign(df['Midday'] - df['open'])
    df['PM_Fragmentation_Sign'] = np.sign(df['close'] - df['Midday'])
    df['Fragmentation_Alignment'] = df['AM_Fragmentation_Sign'] * df['PM_Fragmentation_Sign']
    
    # Fragmentation magnitude divergence
    df['Fragmentation_Magnitude_Divergence'] = abs(df['AM_Fragmentation'] - df['PM_Fragmentation'])
    
    # Gap-fragmentation regime
    df['Gap_Fragmentation_Regime'] = df['Gap_Fragmentation_Momentum'] * df['Fragmentation_Alignment']
    
    # Assume AM and PM amounts are half of daily amount for simplicity
    df['AM_Amount'] = df['amount'] * 0.5
    df['PM_Amount'] = df['amount'] * 0.5
    
    # Liquidity fragmentation
    df['AM_Liquidity_Fragmentation'] = df['AM_Amount'] / (df['AM_Fragmentation'] + 1e-8)
    df['PM_Liquidity_Fragmentation'] = df['PM_Amount'] / (df['PM_Fragmentation'] + 1e-8)
    df['Liquidity_Fragmentation_Divergence'] = df['AM_Liquidity_Fragmentation'] - df['PM_Liquidity_Fragmentation']
    
    # Range-liquidity efficiency
    df['AM_Range_Liquidity_Efficiency'] = df['AM_Fragmentation'] / (df['AM_Amount'] / (df['volume'] + 1e-8) + 1e-8)
    df['PM_Range_Liquidity_Efficiency'] = df['PM_Fragmentation'] / (df['PM_Amount'] / (df['volume'] + 1e-8) + 1e-8)
    df['Range_Liquidity_Fragmentation'] = df['AM_Range_Liquidity_Efficiency'] - df['PM_Range_Liquidity_Efficiency']
    
    # Volume-fragmentation correlation (3-day rolling)
    df['Volume_Fragmentation_Correlation'] = df.groupby(level=1).apply(
        lambda x: x['volume'].rolling(window=3, min_periods=1).corr(x['Gap_Fragmentation_Momentum'])
    ).reset_index(level=1, drop=True)
    
    # Amount-confirmation fragmentation
    df['Amount_Confirmation_Fragmentation'] = df['Liquidity_Fragmentation_Divergence'] * df['Range_Liquidity_Fragmentation']
    
    # Liquidity regime alignment
    df['Volume_3d_Avg'] = df.groupby(level=1)['volume'].rolling(window=3, min_periods=1).mean().reset_index(level=1, drop=True)
    df['Liquidity_Regime_Alignment'] = (df['volume'] / (df['Volume_3d_Avg'] + 1e-8)) * df['Fragmentation_Alignment']
    
    # Morning session fragmentation
    df['Morning_Range_Fragmentation'] = df['AM_Fragmentation'] / (df['AM_Range'] + 1e-8)
    df['Morning_Gap_Fragmentation'] = (df['Midday'] - df['open']) / ((df['open'] - df['PrevClose']) + 1e-8)
    df['Morning_Liquidity_Concentration'] = (df['volume'] * 0.5) / (df['volume'] + 1e-8)
    
    # Afternoon session efficiency
    df['Afternoon_Range_Fragmentation'] = df['PM_Fragmentation'] / (df['PM_Range'] + 1e-8)
    df['Gap_Filling_Fragmentation'] = (df['close'] - df['Midday']) / ((df['open'] - df['PrevClose']) + 1e-8)
    df['Afternoon_Liquidity_Acceleration'] = (df['volume'] * 0.5) / ((df['volume'] * 0.5) + 1e-8)
    
    # Session transition fragmentation
    df['AM_PM_Efficiency_Fragmentation'] = df['Morning_Range_Fragmentation'] - df['Afternoon_Range_Fragmentation']
    df['Liquidity_Transition_Fragmentation'] = df['Morning_Liquidity_Concentration'] * df['Afternoon_Liquidity_Acceleration']
    df['Session_Gap_Fragmentation'] = df['Morning_Gap_Fragmentation'] * df['Gap_Filling_Fragmentation']
    
    # Combined fragmentation-efficiency framework
    df['Base_Fragmentation'] = df['Gap_Fragmentation_Momentum'] * df['Fragmentation_Magnitude_Divergence']
    df['Liquidity_Confirmation'] = df['Base_Fragmentation'] * df['Liquidity_Regime_Alignment']
    df['Range_Liquidity_Validation'] = df['Liquidity_Confirmation'] * df['Amount_Confirmation_Fragmentation']
    
    df['AM_PM_Fragmentation'] = df['Range_Liquidity_Validation'] * df['AM_PM_Efficiency_Fragmentation']
    df['Liquidity_Transition'] = df['AM_PM_Fragmentation'] * df['Liquidity_Transition_Fragmentation']
    df['Gap_Session_Fragmentation'] = df['Liquidity_Transition'] * df['Session_Gap_Fragmentation']
    
    df['Fragmentation_Persistence'] = df['Gap_Session_Fragmentation'] * df['Fragmentation_Alignment']
    df['Range_Efficiency_Context'] = df['Fragmentation_Persistence'] * df['Range_Efficiency_Fragmentation']
    df['Gap_Fragmentation_Final'] = df['Range_Efficiency_Context'] * df['Gap_Fragmentation_Momentum']
    
    # Regime-adaptive fragmentation construction
    df['Short_Term_Fragmentation_Regime'] = df.groupby(level=1)['Gap_Fragmentation_Momentum'].rolling(window=3, min_periods=1).mean().reset_index(level=1, drop=True)
    df['Liquidity_Fragmentation_Context'] = df.groupby(level=1)['Liquidity_Fragmentation_Divergence'].rolling(window=3, min_periods=1).mean().reset_index(level=1, drop=True)
    df['Range_Liquidity_Environment'] = df.groupby(level=1)['Range_Liquidity_Fragmentation'].rolling(window=3, min_periods=1).mean().reset_index(level=1, drop=True)
    
    df['Range_Fragmentation_Regime'] = df.groupby(level=1)['Range_Utilization_Divergence'].rolling(window=5, min_periods=1).mean().reset_index(level=1, drop=True)
    df['Gap_Fragmentation_Environment'] = df.groupby(level=1)['Gap_Momentum_Fragmentation'].rolling(window=5, min_periods=1).std().reset_index(level=1, drop=True)
    df['Liquidity_Concentration_Context'] = df.groupby(level=1)['Morning_Liquidity_Concentration'].rolling(window=5, min_periods=1).mean().reset_index(level=1, drop=True)
    
    # Adaptive weighting
    df['Fragmentation_Regime_Weighting'] = df['Gap_Fragmentation_Final'] * df['Short_Term_Fragmentation_Regime']
    df['Liquidity_Environment_Scaling'] = df['Fragmentation_Regime_Weighting'] * df['Liquidity_Fragmentation_Context']
    df['Efficiency_Context_Final'] = df['Liquidity_Environment_Scaling'] * df['Range_Fragmentation_Regime']
    
    # Final alpha factor
    alpha_factor = df.groupby(level=0)['Efficiency_Context_Final'].last()
    
    return alpha_factor
