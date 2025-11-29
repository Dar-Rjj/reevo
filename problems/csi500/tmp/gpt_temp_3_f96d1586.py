import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Need enough data for rolling calculations
            factor.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        # 1. Intraday Momentum Efficiency Divergence (IMED)
        # Calculate intraday momentum components
        upward_potential = current_data['high'] - current_data['open']
        upward_actual = current_data['close'] - current_data['open']
        downward_potential = current_data['open'] - current_data['low']
        downward_actual = current_data['open'] - current_data['close']
        
        # Momentum utilization ratios
        up_utilization = np.where(upward_potential != 0, 
                                 np.abs(upward_actual) / np.abs(upward_potential), 0)
        down_utilization = np.where(downward_potential != 0, 
                                   np.abs(downward_actual) / np.abs(downward_potential), 0)
        
        # Efficiency divergence (5d vs 20d rolling)
        up_util_5d = up_utilization[-5:].mean()
        up_util_20d = up_utilization[-20:].mean()
        down_util_5d = down_utilization[-5:].mean()
        down_util_20d = down_utilization[-20:].mean()
        
        imed_signal = (up_util_5d - up_util_20d) - (down_util_5d - down_util_20d)
        
        # 2. Range-Volume Convergence Factor (RVCF)
        # Price range analysis
        price_range = current_data['high'] - current_data['low']
        range_efficiency = np.where(price_range != 0, 
                                  np.abs(current_data['close'] - current_data['open']) / price_range, 0)
        
        # Range acceleration (3-day trend)
        range_accel = range_efficiency[-3:].mean() - range_efficiency[-6:-3].mean()
        
        # Volume integration
        volume_trend = current_data['volume'][-3:].mean() / current_data['volume'][-6:-3].mean()
        rvcf_score = range_accel * volume_trend
        
        # 3. Volatility-Adjusted Gap Momentum (VAGM)
        if i > 0:
            # Overnight gap analysis
            gap = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / current_data['close'].iloc[-2]
            
            # Volatility context (5-day average true range)
            high_low = current_data['high'][-5:] - current_data['low'][-5:]
            high_close = np.abs(current_data['high'][-5:] - current_data['close'].shift(1)[-5:])
            low_close = np.abs(current_data['low'][-5:] - current_data['close'].shift(1)[-5:])
            true_ranges = np.maximum(np.maximum(high_low, high_close), low_close)
            atr = true_ranges.mean()
            
            # Gap persistence
            intraday_return = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / current_data['open'].iloc[-1]
            gap_persistence = intraday_return / gap if gap != 0 else 0
            
            vagm_signal = gap_persistence * (current_data['volume'].iloc[-1] / current_data['volume'][-5:].mean())
        else:
            vagm_signal = 0
        
        # 4. Price-Volume Acceleration Divergence (PVAD)
        # Price momentum
        returns_1d = current_data['close'].pct_change(1)[-1]
        returns_2d = (current_data['close'].iloc[-1] / current_data['close'].iloc[-3] - 1) if i >= 2 else 0
        returns_3d = (current_data['close'].iloc[-1] / current_data['close'].iloc[-4] - 1) if i >= 3 else 0
        
        price_accel = (returns_1d + returns_2d/2 + returns_3d/3) / 3
        
        # Volume acceleration
        volume_accel = current_data['volume'][-3:].mean() / current_data['volume'][-6:-3].mean()
        
        pvad_signal = price_accel - volume_accel
        
        # 5. Intraday Reversal Efficiency Score (IRES)
        # Reversal pattern identification
        high_extreme = (current_data['high'] - current_data['open']) / current_data['open']
        low_extreme = (current_data['open'] - current_data['low']) / current_data['open']
        
        # Close position relative to extremes
        close_position = np.where(
            current_data['close'] > current_data['open'],
            (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['open']),
            (current_data['open'] - current_data['close']) / (current_data['open'] - current_data['low'])
        )
        
        # Reversal completion percentage
        reversal_efficiency = np.nan_to_num(close_position, nan=0.5)
        
        # Volume-weighted efficiency score
        current_volume = current_data['volume'].iloc[-1]
        avg_volume = current_data['volume'][-20:].mean()
        ires_score = reversal_efficiency[-1] * (current_volume / avg_volume)
        
        # Combine all factors with equal weights
        factor.iloc[i] = (imed_signal + rvcf_score + vagm_signal + pvad_signal + ires_score) / 5
    
    return factor
