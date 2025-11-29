import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining intraday efficiency, volume dynamics, 
    multi-timeframe volatility, and overnight reversal patterns.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        current_data = df.loc[:date].copy()
        
        if len(current_data) < 10:  # Minimum data requirement
            result.loc[date] = 0
            continue
            
        # 1. Intraday Price Efficiency
        # Opening Gap Momentum
        if len(current_data) >= 2:
            prev_close = current_data['close'].iloc[-2]
            current_open = current_data['open'].iloc[-1]
            current_high = current_data['high'].iloc[-1]
            current_low = current_data['low'].iloc[-1]
            
            gap_ratio = (current_open - prev_close) / prev_close
            intraday_range = (current_high - current_low) / current_open
            
            gap_momentum = gap_ratio / (intraday_range + 1e-8)
        else:
            gap_momentum = 0
            
        # Price Path Optimality
        if len(current_data) >= 2:
            actual_movement = abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1])
            min_movement = abs(current_data['high'].iloc[-1] - current_data['low'].iloc[-1])
            efficiency_coef = actual_movement / (min_movement + 1e-8)
        else:
            efficiency_coef = 0
            
        intraday_efficiency = gap_momentum * efficiency_coef
        
        # 2. Volume-Price Dynamics
        # Amount-Based Impact
        if len(current_data) >= 5:
            price_change = current_data['close'].pct_change().iloc[-1]
            amount = current_data['amount'].iloc[-1]
            price_per_amount = price_change / (amount + 1e-8)
            
            # Deviation from rolling median
            rolling_median = current_data['amount'].rolling(window=5, min_periods=3).median().iloc[-1]
            amount_deviation = amount / (rolling_median + 1e-8)
            amount_impact = price_per_amount * amount_deviation
        else:
            amount_impact = 0
            
        # Volume Elasticity Anomalies
        if len(current_data) >= 6:
            volume_change = current_data['volume'].pct_change().iloc[-1]
            price_change_vol = current_data['close'].pct_change().iloc[-1]
            
            # Calculate rolling elasticity
            vol_changes = current_data['volume'].pct_change().iloc[-5:]
            price_changes = current_data['close'].pct_change().iloc[-5:]
            valid_mask = (vol_changes != 0) & ~vol_changes.isna() & ~price_changes.isna()
            
            if valid_mask.sum() > 2:
                rolling_elasticity = np.median(price_changes[valid_mask] / vol_changes[valid_mask])
                current_elasticity = price_change_vol / (volume_change + 1e-8)
                elasticity_anomaly = current_elasticity - rolling_elasticity
            else:
                elasticity_anomaly = 0
        else:
            elasticity_anomaly = 0
            
        volume_dynamics = amount_impact + elasticity_anomaly
        
        # 3. Multi-Timeframe Volatility
        # Range Convergence
        if len(current_data) >= 6:
            ranges = {}
            for window in [1, 3, 5]:
                if len(current_data) >= window:
                    recent_data = current_data.tail(window)
                    high_low_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].iloc[0]
                    ranges[window] = high_low_range
            
            # Volatility ratio consistency
            if len(ranges) >= 2:
                range_ratios = []
                keys = list(ranges.keys())
                for i in range(len(keys)-1):
                    ratio = ranges[keys[i+1]] / (ranges[keys[i]] + 1e-8)
                    range_ratios.append(ratio)
                
                range_convergence = np.std(range_ratios) if range_ratios else 0
            else:
                range_convergence = 0
        else:
            range_convergence = 0
            
        # Compression Breakout Signals
        if len(current_data) >= 10:
            recent_volatility = current_data['close'].pct_change().rolling(window=5, min_periods=3).std().iloc[-1]
            long_term_volatility = current_data['close'].pct_change().rolling(window=10, min_periods=5).std().iloc[-1]
            
            if long_term_volatility > 0:
                compression_ratio = recent_volatility / long_term_volatility
                breakout_signal = 1 - compression_ratio  # Higher when compressed
            else:
                breakout_signal = 0
        else:
            breakout_signal = 0
            
        volatility_factors = range_convergence + breakout_signal
        
        # 4. Overnight Reversal Patterns
        # Close-Open Return Extremes
        if len(current_data) >= 3:
            overnight_return = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / current_data['close'].iloc[-2]
            intraday_high = current_data['high'].iloc[-1]
            intraday_low = current_data['low'].iloc[-1]
            current_close = current_data['close'].iloc[-1]
            
            # Compare with intraday extremes
            high_deviation = (intraday_high - current_open) / current_open
            low_deviation = (current_open - intraday_low) / current_open
            
            extreme_comparison = abs(overnight_return) / (max(abs(high_deviation), abs(low_deviation)) + 1e-8)
        else:
            extreme_comparison = 0
            
        # Reversal Strength Scoring
        if len(current_data) >= 4:
            prev_overnight = (current_data['open'].iloc[-2] - current_data['close'].iloc[-3]) / current_data['close'].iloc[-3]
            prev_intraday = (current_data['close'].iloc[-2] - current_data['open'].iloc[-2]) / current_data['open'].iloc[-2]
            
            if prev_overnight != 0:
                reversal_intensity = -prev_intraday / (prev_overnight + 1e-8)
            else:
                reversal_intensity = 0
        else:
            reversal_intensity = 0
            
        reversal_patterns = extreme_comparison + reversal_intensity
        
        # Combine all factors
        factor_value = (
            intraday_efficiency * 0.25 +
            volume_dynamics * 0.25 +
            volatility_factors * 0.25 +
            reversal_patterns * 0.25
        )
        
        result.loc[date] = factor_value
    
    return result
