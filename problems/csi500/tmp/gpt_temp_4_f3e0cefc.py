import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['gap_size'] = (data['open'] - data['prev_close']).abs()
    data['daily_range'] = data['high'] - data['low']
    data['close_open_diff'] = (data['close'] - data['open']).abs()
    
    # Estimate first hour range (using first 1/6.5 of trading day as proxy)
    data['first_hour_high'] = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x.max() if len(x) == 5 else np.nan)
    data['first_hour_low'] = data['low'].rolling(window=5, min_periods=1).apply(lambda x: x.min() if len(x) == 5 else np.nan)
    data['first_hour_range'] = data['first_hour_high'] - data['first_hour_low']
    
    # Gap sustainability with range context
    data['gap_range_ratio'] = data['gap_size'] / (data['first_hour_range'] + 1e-8)
    data['gap_sustainability'] = np.exp(-data['gap_range_ratio'])
    
    # Range efficiency
    data['range_efficiency'] = data['close_open_diff'] / (data['daily_range'] + 1e-8)
    data['gap_adjusted_efficiency'] = data['range_efficiency'] * data['gap_sustainability']
    
    # Range efficiency momentum
    data['eff_5d_avg'] = data['range_efficiency'].rolling(window=5, min_periods=3).mean()
    data['eff_momentum'] = data['range_efficiency'] / (data['eff_5d_avg'] + 1e-8)
    data['gap_eff_momentum'] = data['eff_momentum'] * data['gap_sustainability']
    
    # Volume analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_intensity'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    
    # Amount efficiency with gap context
    data['amount_efficiency'] = data['close_open_diff'] / (data['amount'] + 1e-8)
    data['gap_amount_efficiency'] = data['amount_efficiency'] * data['gap_sustainability']
    
    # Range momentum with volume confirmation
    data['range_5d_change'] = data['daily_range'].pct_change(periods=5)
    data['volume_5d_change'] = data['volume'].pct_change(periods=5)
    data['range_volume_momentum'] = data['range_5d_change'] * data['volume_5d_change']
    
    # Closing position strength
    data['close_position'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    data['close_position_momentum'] = data['close_position'].pct_change(periods=3)
    
    # Intraday pressure accumulation (simplified)
    data['intraday_pressure'] = ((data['close'] - data['open']) / (data['daily_range'] + 1e-8)).rolling(window=3, min_periods=2).sum()
    data['gap_pressure'] = data['intraday_pressure'] * data['gap_sustainability']
    
    # Volume-efficiency correlation
    data['vol_eff_corr'] = data['volume_intensity'] * data['range_efficiency']
    data['gap_vol_eff'] = data['vol_eff_corr'] * data['gap_sustainability']
    
    # Volatility cycle detection
    data['range_std_5d'] = data['daily_range'].rolling(window=5, min_periods=3).std()
    data['range_compression'] = data['range_std_5d'] / (data['range_std_5d'].rolling(window=10, min_periods=5).mean() + 1e-8)
    
    # Composite factor calculation
    data['gap_coherence'] = (
        data['gap_sustainability'] * 
        data['gap_eff_momentum'] * 
        data['volume_intensity']
    )
    
    data['range_efficiency_component'] = (
        data['gap_adjusted_efficiency'] * 
        data['range_volume_momentum'] * 
        data['close_position_momentum']
    )
    
    data['pressure_volume_alignment'] = (
        data['gap_pressure'] * 
        data['gap_vol_eff'] * 
        (1 - data['range_compression'])
    )
    
    # Final composite alpha factor
    data['alpha_factor'] = (
        data['gap_coherence'] * 
        data['range_efficiency_component'] * 
        data['pressure_volume_alignment'] * 
        data['gap_amount_efficiency']
    )
    
    # Apply cross-sectional ranking
    def cross_sectional_rank(group):
        return group.rank(pct=True)
    
    data['final_alpha'] = data.groupby(data.index)['alpha_factor'].transform(cross_sectional_rank)
    
    return data['final_alpha']
