import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Required calculations for all dates
    # Daily returns
    data['return'] = data['close'] / data['close'].shift(1) - 1
    
    # True Range
    hl = data['high'] - data['low']
    hc = abs(data['high'] - data['close'].shift(1))
    lc = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = np.maximum(hl, np.maximum(hc, lc))
    
    # High-Low Range
    data['hl_range'] = data['high'] - data['low']
    
    # Typical Price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Money Flow
    data['money_flow'] = data['typical_price'] * data['volume']
    
    # Rolling averages for various windows
    for window in [5, 10, 20]:
        data[f'volume_ma_{window}'] = data['volume'].rolling(window=window, min_periods=1).mean()
        data[f'hl_range_ma_{window}'] = data['hl_range'].rolling(window=window, min_periods=1).mean()
        data[f'close_ma_{window}'] = data['close'].rolling(window=window, min_periods=1).mean()
    
    # Calculate factor for each date
    for i, date in enumerate(data.index):
        if i < 1:  # Skip first day due to shift operations
            factor.loc[date] = 0
            continue
            
        current_data = data.loc[:date].copy()
        
        # 1. High-Low Breakout Persistence
        breakout_count = 0
        consecutive_days = 0
        for j in range(max(0, i-10), i+1):  # Look back up to 10 days
            if j > 0 and current_data.iloc[j]['close'] > current_data.iloc[j-1]['high']:
                consecutive_days += 1
            else:
                consecutive_days = 0
            if consecutive_days > 0:
                breakout_count += consecutive_days * current_data.iloc[j]['hl_range']
        
        hl_breakout_factor = breakout_count / (current_data.iloc[max(0, i-9):i+1]['hl_range'].mean() + 1e-8)
        
        # 2. Volatility-Adjusted Return Momentum
        vol_adj_return = 0
        valid_days = 0
        for j in range(max(0, i-5), i+1):  # 5-day momentum
            if j > 0 and current_data.iloc[j]['true_range'] > 0:
                vol_adj_return += current_data.iloc[j]['return'] / current_data.iloc[j]['true_range']
                valid_days += 1
        
        vol_momentum_factor = vol_adj_return / (valid_days + 1e-8) if valid_days > 0 else 0
        
        # 3. Volume-Adjusted Gap Reversal
        if i > 0:
            price_gap = current_data.iloc[i]['open'] / current_data.iloc[i-1]['close'] - 1
            volume_ratio = current_data.iloc[i]['volume'] / current_data.iloc[i]['volume_ma_10']
            
            if abs(price_gap) > 0.02:  # Large gap
                if volume_ratio < 0.8:  # Low volume
                    gap_factor = -price_gap  # Expect reversal
                else:
                    gap_factor = price_gap  # Expect continuation
            else:  # Small gap
                if volume_ratio > 1.2:  # High volume
                    gap_factor = price_gap  # Expect continuation
                else:
                    gap_factor = -price_gap  # Expect reversal
        else:
            gap_factor = 0
        
        # 4. Intraday Momentum Divergence
        if i > 0:
            intraday_strength = (current_data.iloc[i]['close'] - current_data.iloc[i]['low']) / (current_data.iloc[i]['hl_range'] + 1e-8)
            
            # Simplified volume pattern (using rolling averages as proxy for intraday patterns)
            recent_volume_avg = current_data.iloc[max(0, i-4):i+1]['volume'].mean()
            volume_trend = current_data.iloc[i]['volume'] / (recent_volume_avg + 1e-8)
            
            if intraday_strength > 0.6 and volume_trend < 0.9:  # Strong close + declining volume
                intraday_factor = -1
            elif intraday_strength < 0.4 and volume_trend > 1.1:  # Weak close + rising volume
                intraday_factor = 1
            else:
                intraday_factor = 0
        else:
            intraday_factor = 0
        
        # 5. Money Flow Oscillator
        money_flow_osc = 0
        lookback = min(14, i+1)  # 14-day oscillator
        positive_mf = 0
        negative_mf = 0
        
        for j in range(max(0, i-lookback+1), i+1):
            if j > max(0, i-lookback+1):
                mf_change = current_data.iloc[j]['money_flow'] - current_data.iloc[j-1]['money_flow']
                if mf_change > 0:
                    positive_mf += mf_change
                else:
                    negative_mf += abs(mf_change)
        
        if positive_mf + negative_mf > 0:
            money_flow_osc = (positive_mf - negative_mf) / (positive_mf + negative_mf)
        
        # 6. Price-Volume Congestion Break
        congestion_factor = 0
        recent_volatility = current_data.iloc[max(0, i-9):i+1]['hl_range'].std()
        avg_volatility = current_data.iloc[max(0, i-49):i+1]['hl_range'].std() if i >= 49 else recent_volatility
        
        if recent_volatility < avg_volatility * 0.7:  # Low volatility period
            recent_high = current_data.iloc[max(0, i-9):i+1]['high'].max()
            recent_low = current_data.iloc[max(0, i-9):i+1]['low'].min()
            recent_volume_avg = current_data.iloc[max(0, i-9):i+1]['volume'].mean()
            
            price_break = (current_data.iloc[i]['close'] - recent_low) / (recent_high - recent_low + 1e-8)
            volume_break = current_data.iloc[i]['volume'] / (recent_volume_avg + 1e-8)
            
            if price_break > 0.8 and volume_break > 1.2:  # Breakout with volume confirmation
                congestion_factor = 1
            elif price_break < 0.2 and volume_break > 1.2:  # Breakdown with volume confirmation
                congestion_factor = -1
        
        # Combine all factors with equal weights
        combined_factor = (
            hl_breakout_factor * 0.2 +
            vol_momentum_factor * 0.2 +
            gap_factor * 0.15 +
            intraday_factor * 0.15 +
            money_flow_osc * 0.15 +
            congestion_factor * 0.15
        )
        
        factor.loc[date] = combined_factor
    
    # Normalize the factor
    factor = (factor - factor.mean()) / (factor.std() + 1e-8)
    
    return factor
