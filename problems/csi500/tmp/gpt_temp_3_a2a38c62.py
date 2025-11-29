import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Gap-Momentum Fragmentation with Volume-Amount Synchronization factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price components
    data['prev_close'] = data['close'].shift(1)
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Gap-Momentum Fragmentation Analysis
    data['gap_size'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['am_momentum'] = (data['mid_price'] - data['open']) / data['open']
    data['pm_momentum'] = (data['close'] - data['mid_price']) / data['mid_price']
    
    # Fragmentation Ratio: Gap vs AM-PM Momentum Divergence
    data['momentum_divergence'] = np.abs(data['am_momentum'] - data['pm_momentum'])
    data['fragmentation_ratio'] = np.abs(data['gap_size']) / (data['momentum_divergence'] + 1e-8)
    
    # Volume-Amount Synchronization Assessment
    data['am_volume_momentum_corr'] = data['am_momentum'].rolling(window=5).corr(data['volume'].rolling(window=5).mean())
    data['pm_volume_momentum_corr'] = data['pm_momentum'].rolling(window=5).corr(data['volume'].rolling(window=5).mean())
    
    # Amount per Volume Ratio (AM vs PM)
    data['amount_per_volume'] = data['amount'] / (data['volume'] + 1e-8)
    data['am_amount_efficiency'] = data['amount_per_volume'].rolling(window=3).mean()
    data['pm_amount_efficiency'] = data['amount_per_volume'].rolling(window=3, min_periods=1).apply(
        lambda x: x.iloc[-1] if len(x) > 0 else np.nan
    )
    
    # Volume-Amount Co-movement During Gap Events
    gap_threshold = data['gap_size'].abs().rolling(window=10).quantile(0.7)
    data['large_gap_flag'] = (data['gap_size'].abs() > gap_threshold).astype(int)
    data['gap_volume_amount_sync'] = data['large_gap_flag'] * data['am_volume_momentum_corr'] * data['pm_volume_momentum_corr']
    
    # Fragmentation-Synchronization Composite
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['gap_fragmentation_range'] = data['fragmentation_ratio'] * data['range_efficiency']
    
    # Volume Synchronization Weighting
    data['volume_sync_weight'] = (data['am_volume_momentum_corr'] + data['pm_volume_momentum_corr']) / 2
    data['amount_efficiency_scaling'] = (data['am_amount_efficiency'] + data['pm_amount_efficiency']) / 2
    
    # Initial composite
    data['frag_sync_composite'] = (data['gap_fragmentation_range'] * 
                                  data['volume_sync_weight'] * 
                                  data['amount_efficiency_scaling'])
    
    # Multi-Timeframe Persistence Analysis
    data['frag_5day_pattern'] = data['frag_sync_composite'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    data['frag_10day_trend'] = data['frag_sync_composite'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    
    # Timeframe-weighted composite
    data['multi_timeframe_composite'] = (0.6 * data['frag_sync_composite'] + 
                                       0.25 * data['frag_5day_pattern'] + 
                                       0.15 * data['frag_10day_trend'])
    
    # Final Alpha Factor Generation
    # Bounded Signal Transformation
    rolling_std = data['multi_timeframe_composite'].rolling(window=20, min_periods=10).std()
    rolling_mean = data['multi_timeframe_composite'].rolling(window=20, min_periods=10).mean()
    
    data['bounded_signal'] = (data['multi_timeframe_composite'] - rolling_mean) / (rolling_std + 1e-8)
    data['bounded_signal'] = np.tanh(data['bounded_signal'] * 0.5)  # Soft bounding
    
    # Persistence Validation
    data['signal_persistence'] = data['bounded_signal'].rolling(window=5).std()
    data['validated_signal'] = data['bounded_signal'] / (data['signal_persistence'] + 1e-8)
    
    # Raw Composite Output
    alpha_factor = data['validated_signal']
    
    return alpha_factor
