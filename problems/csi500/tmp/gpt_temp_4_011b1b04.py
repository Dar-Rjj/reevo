import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency Analysis
    # Intraday efficiency
    data['intraday_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_eff'] = data['intraday_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Overnight efficiency
    data['prev_close'] = data['close'].shift(1)
    data['overnight_eff'] = np.abs(data['open'] - data['prev_close']) / (data['high'] - data['low'])
    data['overnight_eff'] = data['overnight_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Efficiency regime classification
    intraday_eff_rank = data['intraday_eff'].rolling(window=5, min_periods=3).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    overnight_eff_rank = data['overnight_eff'].rolling(window=5, min_periods=3).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    data['eff_regime'] = 0
    data.loc[(intraday_eff_rank > 0.7) & (overnight_eff_rank < 0.3), 'eff_regime'] = 1  # directional trending
    data.loc[(intraday_eff_rank < 0.3) & (overnight_eff_rank > 0.7), 'eff_regime'] = -1  # gap-driven
    data['eff_regime'] = data['eff_regime'].rolling(window=3, min_periods=2).mean()
    
    # Price Anchoring Strength
    # Opening anchoring
    data['open_anchor'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['open_anchor'] = data['open_anchor'].replace([np.inf, -np.inf], np.nan)
    
    # Midpoint anchoring persistence
    midpoint = (data['high'] + data['low']) / 2
    data['midpoint_proximity'] = 1 - (np.abs(data['close'] - midpoint) / (data['high'] - data['low']))
    data['midpoint_proximity'] = data['midpoint_proximity'].replace([np.inf, -np.inf], np.nan)
    data['anchor_strength'] = (data['open_anchor'].rolling(window=5, min_periods=3).mean() + 
                              data['midpoint_proximity'].rolling(window=5, min_periods=3).mean()) / 2
    
    # Anchoring-efficiency interaction
    data['anchor_eff_interaction'] = data['anchor_strength'] * data['intraday_eff']
    
    # Volume-Efficiency Elasticity
    data['volume_elasticity'] = data['volume'] / (data['high'] - data['low'])
    data['volume_elasticity'] = data['volume_elasticity'].replace([np.inf, -np.inf], np.nan)
    
    # Elasticity-anchoring confirmation
    vol_elasticity_rank = data['volume_elasticity'].rolling(window=5, min_periods=3).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    anchor_strength_rank = data['anchor_strength'].rolling(window=5, min_periods=3).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    data['elasticity_confirmation'] = 0
    data.loc[(vol_elasticity_rank > 0.7) & (anchor_strength_rank > 0.7), 'elasticity_confirmation'] = 1
    data.loc[(vol_elasticity_rank > 0.7) & (anchor_strength_rank < 0.3), 'elasticity_confirmation'] = -1
    
    # Microstructure Fracture Detection
    # Price fracture
    data['prev_high_low_range'] = (data['high'] - data['low']).shift(1)
    data['price_fracture'] = (data['high'] - data['low']) / data['prev_high_low_range']
    data['price_fracture'] = data['price_fracture'].replace([np.inf, -np.inf], np.nan)
    
    # Volume fracture
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['volume_fracture'] = (data['volume'] / data['prev_volume']) * (data['amount'] / data['prev_amount'])
    data['volume_fracture'] = data['volume_fracture'].replace([np.inf, -np.inf], np.nan)
    
    # Opening-Closing Regime Analysis
    data['open_regime'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['close_regime'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['regime_divergence'] = data['open_regime'] - data['close_regime']
    
    # Composite Alpha Generation
    # Dynamic signal integration
    eff_weighted = data['intraday_eff'] * data['anchor_strength']
    vol_enhanced = eff_weighted * (1 + data['volume_elasticity'].rolling(window=5, min_periods=3).mean())
    
    # Incorporate microstructure signals
    price_fracture_signal = np.sign(data['price_fracture'] - 1) * np.log(np.abs(data['price_fracture']))
    volume_fracture_signal = np.sign(data['volume_fracture'] - 1) * np.log(np.abs(data['volume_fracture']))
    
    microstructure_component = (price_fracture_signal + volume_fracture_signal) / 2
    
    # Final composite factor
    composite_factor = (vol_enhanced * (1 + data['elasticity_confirmation']) + 
                       microstructure_component * data['regime_divergence'])
    
    # Rolling normalization
    composite_factor_ma = composite_factor.rolling(window=10, min_periods=5).mean()
    composite_factor_std = composite_factor.rolling(window=10, min_periods=5).std()
    normalized_factor = (composite_factor - composite_factor_ma) / composite_factor_std
    
    return normalized_factor
