import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Required minimum period for calculations
    min_period = 20
    
    for i in range(min_period, len(data)):
        current_data = data.iloc[:i+1].copy()
        
        # Volatility-Regime Fractal Momentum Divergence
        # Fractal Momentum Components
        close_prices = current_data['close'].values
        
        # Short-Term Fractal Efficiency (3-day)
        if i >= 3:
            short_term_numerator = abs(close_prices[i] - close_prices[i-3])
            short_term_denominator = sum(abs(close_prices[j] - close_prices[j-1]) for j in range(max(1, i-2), i+1))
            short_term_fe = short_term_numerator / short_term_denominator if short_term_denominator != 0 else 0
        else:
            short_term_fe = 0
            
        # Long-Term Fractal Efficiency (15-day)
        if i >= 15:
            long_term_numerator = abs(close_prices[i] - close_prices[i-15])
            long_term_denominator = sum(abs(close_prices[j] - close_prices[j-1]) for j in range(max(1, i-14), i+1))
            long_term_fe = long_term_numerator / long_term_denominator if long_term_denominator != 0 else 0
        else:
            long_term_fe = 0
            
        fractal_momentum_divergence = short_term_fe - long_term_fe
        
        # Volatility-Regime Integration
        current_row = current_data.iloc[-1]
        prev_close = close_prices[i-1] if i > 0 else current_row['close']
        
        intraday_vol = (current_row['high'] - current_row['low']) / abs(current_row['close'] - current_row['open']) if abs(current_row['close'] - current_row['open']) != 0 else 0
        gap_vol = abs(current_row['open'] - prev_close) / (current_row['high'] - current_row['low']) if (current_row['high'] - current_row['low']) != 0 else 0
        vol_vol_alignment = current_row['volume'] * (current_row['high'] - current_row['low']) / abs(current_row['close'] - current_row['open']) if abs(current_row['close'] - current_row['open']) != 0 else 0
        
        volatility_regime_component = fractal_momentum_divergence * (intraday_vol + gap_vol + vol_vol_alignment) / 3
        
        # Efficiency-Persistence Gap Asymmetry
        # Gap Asymmetry Components
        opening_gap_dir = np.sign(current_row['open'] - prev_close)
        
        # Gap Magnitude Persistence
        gap_persistence = 0
        for j in range(min(5, i), 0, -1):
            if j == i:
                continue
            prev_gap_dir = np.sign(current_data.iloc[j]['open'] - current_data.iloc[j-1]['close'])
            if prev_gap_dir == opening_gap_dir:
                gap_persistence += 1
            else:
                break
        
        # Gap Efficiency Ratio
        daily_range_sum = sum(current_data.iloc[max(0, i-4):i+1]['high'] - current_data.iloc[max(0, i-4):i+1]['low'])
        gap_efficiency = abs(current_row['open'] - prev_close) / daily_range_sum if daily_range_sum != 0 else 0
        
        # Efficiency Persistence Metrics
        intraday_eff_dir = np.sign(current_row['close'] - current_row['open'])
        
        # Efficiency Persistence Streak
        eff_persistence = 0
        for j in range(min(5, i), 0, -1):
            if j == i:
                continue
            prev_eff_dir = np.sign(current_data.iloc[j]['close'] - current_data.iloc[j]['open'])
            if prev_eff_dir == intraday_eff_dir:
                eff_persistence += 1
            else:
                break
        
        efficiency_gap_alignment = opening_gap_dir * intraday_eff_dir
        
        efficiency_persistence_component = (gap_persistence + eff_persistence + efficiency_gap_alignment) * gap_efficiency
        
        # Liquidity-Fractal Breakout Concentration
        # Breakout Concentration Components
        prev_high = current_data.iloc[i-1]['high'] if i > 0 else current_row['high']
        prev_low = current_data.iloc[i-1]['low'] if i > 0 else current_row['low']
        
        fractal_breakout_intensity = short_term_fe * abs(current_row['close'] - prev_close) / prev_close if prev_close != 0 else 0
        
        breakout_detection = 1 if (current_row['close'] > prev_high or current_row['close'] < prev_low) else 0
        volume_breakout_concentration = current_row['volume'] * breakout_detection
        
        liquidity_breakout_ratio = current_row['amount'] / (current_row['high'] - current_row['low']) * breakout_detection if (current_row['high'] - current_row['low']) != 0 else 0
        
        # Multi-Timeframe Integration
        volume_5d_median = np.median(current_data.iloc[max(0, i-4):i+1]['volume']) if i >= 4 else current_row['volume']
        volume_20d_median = np.median(current_data.iloc[max(0, i-19):i+1]['volume']) if i >= 19 else current_row['volume']
        
        short_term_volume_surge = current_row['volume'] / volume_5d_median if volume_5d_median != 0 else 0
        medium_term_volume_trend = current_row['volume'] / volume_20d_median if volume_20d_median != 0 else 0
        
        # 10-day Fractal Efficiency
        if i >= 10:
            fe_10_numerator = abs(close_prices[i] - close_prices[i-10])
            fe_10_denominator = sum(abs(close_prices[j] - close_prices[j-1]) for j in range(max(1, i-9), i+1))
            fe_10 = fe_10_numerator / fe_10_denominator if fe_10_denominator != 0 else 0
        else:
            fe_10 = 0
            
        fractal_efficiency_regime = short_term_fe / fe_10 if fe_10 != 0 else 0
        
        liquidity_component = (fractal_breakout_intensity + volume_breakout_concentration + liquidity_breakout_ratio) * \
                             (short_term_volume_surge + medium_term_volume_trend + fractal_efficiency_regime) / 3
        
        # Combine all components
        factor_value = (volatility_regime_component + 
                       efficiency_persistence_component + 
                       liquidity_component) / 3
        
        result.iloc[i] = factor_value
    
    # Fill initial values with 0
    result = result.fillna(0)
    
    return result
