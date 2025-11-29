import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Price-Volume-Range Congruence Factor with multi-timeframe analysis
    """
    data = df.copy()
    
    # Multi-Timeframe Range Efficiency Analysis
    # Short-term Range Dynamics (3-day)
    data['range'] = data['high'] - data['low']
    data['range_efficiency'] = (data['close'] - data['low']) / (data['range'] + 1e-8)
    data['range_momentum'] = data['range'] / (data['range'].shift(1) + 1e-8) - 1
    
    # Medium-term Range Patterns (5-day)
    data['range_volatility_5d'] = data['range'].rolling(window=5).std()
    data['range_compression'] = data['range'] / data['range'].rolling(window=5).mean() - 1
    data['range_percentile'] = data['range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    
    # Volume-Price Integration Framework
    data['volume_momentum'] = data['volume'] / (data['volume'].shift(1) + 1e-8) - 1
    data['volume_range_ratio'] = data['volume'] / (data['range'] + 1e-8)
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Price-Volume Congruence
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['pv_alignment'] = np.sign(data['price_momentum']) * np.sign(data['volume_momentum'])
    data['volume_weighted_efficiency'] = data['range_efficiency'] * data['volume_concentration']
    
    # Divergence detection
    data['price_volume_divergence'] = (
        data['price_momentum'].rolling(window=3).mean() - 
        data['volume_momentum'].rolling(window=3).mean()
    )
    
    # Market Regime Classification
    data['volatility_regime'] = (data['range_volatility_5d'] > 
                                data['range_volatility_5d'].rolling(window=20).median()).astype(int)
    data['volume_regime'] = (data['volume'] > 
                            data['volume'].rolling(window=20).median()).astype(int)
    data['trend_state'] = (data['close'].rolling(window=5).mean() > 
                          data['close'].rolling(window=20).mean()).astype(int)
    
    # Regime-Specific Congruence Scoring
    # High Volatility regime signals
    high_vol_mask = data['volatility_regime'] == 1
    data['hv_triple_alignment'] = 0
    data.loc[high_vol_mask, 'hv_triple_alignment'] = (
        (data['range_momentum'] > 0) & 
        (data['volume_momentum'] > 0) & 
        (data['price_momentum'] > 0)
    ).astype(int)
    
    data['hv_contraction_reversal'] = 0
    data.loc[high_vol_mask, 'hv_contraction_reversal'] = (
        (data['range_momentum'] < 0) & 
        (data['volume_momentum'] < 0) & 
        (data['price_momentum'] < 0)
    ).astype(int)
    
    # Low Volatility regime signals
    low_vol_mask = data['volatility_regime'] == 0
    data['lv_efficiency_volume'] = 0
    data.loc[low_vol_mask, 'lv_efficiency_volume'] = (
        (data['range_efficiency'] > 0.6) & 
        (data['volume_concentration'] > 1)
    ).astype(int)
    
    data['lv_consolidation'] = 0
    data.loc[low_vol_mask, 'lv_consolidation'] = (
        (data['range_compression'] < -0.2) & 
        (data['volume_concentration'] < 0.8)
    ).astype(int)
    
    # Gap Analysis
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['gap_direction'] = np.sign(data['overnight_gap'])
    data['range_utilization'] = data['range'] / (abs(data['overnight_gap']) * data['close'].shift(1) + 1e-8)
    data['close_gap_position'] = (data['close'] - data['open']) / (abs(data['overnight_gap']) * data['close'].shift(1) + 1e-8)
    
    # Volume-Confirmed Gap Dynamics
    data['opening_volume_intensity'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['volume_gap_ratio'] = data['volume'] / (abs(data['overnight_gap']) + 1e-8)
    
    # Gap signals
    data['gap_confirmation'] = (
        (abs(data['overnight_gap']) > 0.01) & 
        (data['opening_volume_intensity'] > 1.2) & 
        (np.sign(data['close_gap_position']) == np.sign(data['overnight_gap']))
    ).astype(int)
    
    data['gap_reversal'] = (
        (abs(data['overnight_gap']) > 0.01) & 
        (data['opening_volume_intensity'] < 0.8) & 
        (np.sign(data['close_gap_position']) != np.sign(data['overnight_gap']))
    ).astype(int)
    
    # Price-Range-Volume Momentum Triangulation
    data['price_momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['range_momentum_3d'] = data['range'] / data['range'].shift(3).rolling(window=3).mean() - 1
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3).rolling(window=3).mean() - 1
    
    # Vector alignment scoring
    data['triple_alignment'] = (
        np.sign(data['price_momentum_3d']) + 
        np.sign(data['range_momentum_3d']) + 
        np.sign(data['volume_momentum_3d'])
    ).abs()
    
    # Divergence patterns
    data['price_range_divergence'] = abs(
        data['price_momentum_3d'] - data['range_momentum_3d']
    )
    data['price_volume_divergence_3d'] = abs(
        data['price_momentum_3d'] - data['volume_momentum_3d']
    )
    
    # Intraday Momentum Compression-Expansion Cycle
    data['range_compression_3d'] = data['range'] / data['range'].rolling(window=3).mean() - 1
    data['compression_duration'] = data['range_compression'].rolling(window=5).apply(
        lambda x: (x < -0.1).sum()
    )
    
    # Expansion detection
    data['expansion_momentum'] = (
        (data['range_compression_3d'] > 0.1) & 
        (data['volume_concentration'] > 1.1)
    ).astype(int)
    
    # Cycle phase detection
    data['compression_phase'] = (data['range_compression_3d'] < -0.15).astype(int)
    data['expansion_phase'] = (data['range_compression_3d'] > 0.15).astype(int)
    
    # Final signal integration with regime conditioning
    data['final_signal'] = (
        # High volatility components
        data['hv_triple_alignment'] * 0.3 * data['volatility_regime'] +
        data['hv_contraction_reversal'] * -0.2 * data['volatility_regime'] +
        
        # Low volatility components  
        data['lv_efficiency_volume'] * 0.25 * (1 - data['volatility_regime']) +
        data['lv_consolidation'] * -0.15 * (1 - data['volatility_regime']) +
        
        # Gap signals
        data['gap_confirmation'] * 0.2 * np.sign(data['overnight_gap']) +
        data['gap_reversal'] * -0.25 * np.sign(data['overnight_gap']) +
        
        # Triangulation alignment
        data['triple_alignment'] * 0.1 +
        
        # Cycle signals
        data['expansion_phase'] * 0.15 * data['volume_concentration'] +
        data['compression_phase'] * -0.1 * (1 - data['volume_concentration'])
    )
    
    # Normalize final signal
    signal = data['final_signal'].fillna(0)
    
    return signal
