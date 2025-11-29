import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining:
    - Gap Reversal Momentum
    - Range Breakout Efficiency  
    - Volatility Regime Momentum
    - Microstructure Pressure
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate common metrics
    prev_close = df['close'].shift(1)
    daily_range = df['high'] - df['low']
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    for i in range(len(df)):
        if i < 5:  # Need enough history for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        current_day = current_data.iloc[-1]
        prev_data = current_data.iloc[:-1]
        
        # 1. Gap Reversal Momentum
        if i >= 1:
            overnight_gap = current_day['open'] / prev_data['close'].iloc[-1] - 1
            intraday_fade = current_day['close'] / current_day['open'] - 1
            gap_reversal = -overnight_gap * intraday_fade
        else:
            gap_reversal = 0
        
        # 2. Range Breakout Efficiency
        current_range = daily_range.iloc[i]
        avg_5d_range = daily_range.iloc[i-5:i].mean()
        range_expansion = current_range / avg_5d_range - 1 if avg_5d_range > 0 else 0
        
        current_volume = current_day['volume']
        avg_5d_volume = prev_data['volume'].iloc[-5:].mean()
        volume_surge = current_volume / avg_5d_volume - 1 if avg_5d_volume > 0 else 0
        
        range_breakout = range_expansion * volume_surge
        
        # 3. Volatility Regime Momentum
        if i >= 3:
            # 3-day volatility (using daily ranges)
            current_vol = daily_range.iloc[i-2:i+1].std()
            prev_vol = daily_range.iloc[i-5:i-2].std()
            vol_change = current_vol / prev_vol - 1 if prev_vol > 0 else 0
            
            # 3-day price momentum
            price_momentum = current_day['close'] / prev_data['close'].iloc[-3] - 1
            
            vol_momentum = vol_change * price_momentum
            
            # Range persistence (autocorrelation of ranges)
            recent_ranges = daily_range.iloc[i-4:i+1].values
            range_persistence = np.corrcoef(recent_ranges[:-1], recent_ranges[1:])[0,1] if len(recent_ranges) > 1 else 0
            
            # Volume trend (5-day slope)
            volumes = prev_data['volume'].iloc[-5:].values
            if len(volumes) >= 2:
                volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] / np.mean(volumes) if np.mean(volumes) > 0 else 0
            else:
                volume_trend = 0
                
            regime_momentum = vol_momentum * (1 + range_persistence) * (1 + volume_trend)
        else:
            regime_momentum = 0
        
        # 4. Microstructure Pressure
        if i >= 1:
            prev_range = daily_range.iloc[i-1]
            opening_gap_vs_range = abs(overnight_gap) / prev_range if prev_range > 0 else 0
            
            # End-of-day pressure (last hour approximation using close vs typical price)
            eod_pressure = (current_day['close'] - typical_price.iloc[i]) / typical_price.iloc[i]
            
            # Volume concentration (current volume vs 5-day average)
            volume_concentration = volume_surge
            
            microstructure = opening_gap_vs_range * eod_pressure * volume_concentration
        else:
            microstructure = 0
        
        # Combine factors with equal weights
        combined_factor = (
            gap_reversal + 
            range_breakout + 
            regime_momentum + 
            microstructure
        )
        
        result.iloc[i] = combined_factor
    
    return result
