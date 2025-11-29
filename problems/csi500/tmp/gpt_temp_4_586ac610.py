import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for current_date in df.index:
        current_data = df.loc[current_date]
        
        # Skip if insufficient data
        if pd.isna(current_data['close']) or pd.isna(current_data['open']) or \
           pd.isna(current_data['high']) or pd.isna(current_data['low']) or \
           pd.isna(current_data['volume']) or pd.isna(current_data['amount']):
            result.loc[current_date] = np.nan
            continue
        
        # Get previous close safely
        prev_idx = df.index.get_indexer([current_date], method='pad')[0]
        if prev_idx == 0:
            prev_close = np.nan
        else:
            prev_date = df.index[prev_idx - 1]
            prev_close = df.loc[prev_date, 'close']
        
        if pd.isna(prev_close):
            result.loc[current_date] = np.nan
            continue
        
        # Price Path Asymmetry Analysis
        high_low_range = current_data['high'] - current_data['low']
        if high_low_range <= 0:
            intraday_return_asymmetry = np.nan
            close_position_asymmetry = np.nan
        else:
            intraday_return_asymmetry = (current_data['close'] - current_data['open']) / high_low_range
            close_position_asymmetry = (current_data['close'] - current_data['low']) / high_low_range - 0.5
        
        high_low_path_divergence = (current_data['high'] - current_data['open']) - (current_data['open'] - current_data['low'])
        
        # Volume-Price Interaction
        if high_low_range <= 0:
            volume_concentration = np.nan
        else:
            volume_concentration = current_data['volume'] / high_low_range
        
        if current_data['amount'] <= 0:
            price_impact_efficiency = np.nan
        else:
            price_impact_efficiency = abs(current_data['close'] - current_data['open']) / current_data['amount']
        
        if current_data['amount'] <= 0:
            liquidity_absorption = np.nan
        else:
            liquidity_absorption = high_low_range * current_data['volume'] / current_data['amount']
        
        # Time-Weighted Momentum
        early_late_divergence = (current_data['open'] - prev_close) - (current_data['close'] - current_data['open'])
        midday_momentum = (current_data['high'] + current_data['low']) / 2 - current_data['open']
        
        prev_open_close_diff = abs(current_data['open'] - prev_close)
        if prev_open_close_diff <= 0:
            session_persistence = np.nan
        else:
            session_persistence = (current_data['close'] - current_data['open']) / prev_open_close_diff
        
        # Range Efficiency Signals
        if high_low_range <= 0:
            range_utilization = np.nan
            gap_range_alignment = np.nan
        else:
            range_utilization = abs(current_data['close'] - current_data['open']) / high_low_range
            gap_range_alignment = (current_data['open'] - prev_close) / high_low_range
        
        extreme_rejection = (current_data['high'] - current_data['close']) * (current_data['close'] - current_data['low']) * current_data['volume']
        
        # Alpha Synthesis
        core_asymmetry = intraday_return_asymmetry * high_low_path_divergence * close_position_asymmetry
        
        if pd.isna(volume_concentration) or pd.isna(price_impact_efficiency):
            volume_enhancement = core_asymmetry
        else:
            volume_enhancement = core_asymmetry * volume_concentration * price_impact_efficiency
        
        if pd.isna(early_late_divergence) or pd.isna(midday_momentum):
            momentum_integration = volume_enhancement
        else:
            momentum_integration = volume_enhancement * early_late_divergence * midday_momentum
        
        if pd.isna(range_utilization):
            range_filtering = momentum_integration
        else:
            range_filtering = momentum_integration * range_utilization * extreme_rejection
        
        if pd.isna(gap_range_alignment) or pd.isna(liquidity_absorption):
            final_alpha = range_filtering
        else:
            final_alpha = range_filtering * gap_range_alignment * liquidity_absorption
        
        result.loc[current_date] = final_alpha
    
    return result
