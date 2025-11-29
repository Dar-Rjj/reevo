import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Asymmetry Framework for cross-sectional alpha generation
    """
    data = df.copy()
    
    # Volatility Asymmetry Detection
    # Directional Volatility Bias
    data['vol_up'] = data['high'] - data['open']
    data['vol_down'] = data['open'] - data['low']
    data['vol_asymmetry'] = (data['vol_up'] - data['vol_down']) / (data['vol_up'] + data['vol_down'] + 1e-8)
    
    # Multi-day Asymmetry State
    data['vol_asymmetry_5d'] = data['vol_asymmetry'].rolling(window=5, min_periods=3).mean()
    data['asymmetry_state'] = data['vol_asymmetry'] - data['vol_asymmetry_5d']
    
    # Regime Classification
    data['vol_asymmetry_20d_median'] = data['vol_asymmetry'].rolling(window=20, min_periods=10).median()
    data['asymmetry_regime'] = np.where(data['vol_asymmetry'] > data['vol_asymmetry_20d_median'], 1, -1)
    
    # Price-Volume Asymmetry Synthesis
    # Price Asymmetry Acceleration
    data['price_asymmetry'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['price_asymmetry_accel'] = data['price_asymmetry'] - data['price_asymmetry'].shift(3)
    
    # Volume Asymmetry Acceleration
    data['volume_asymmetry'] = data['volume'] * data['vol_asymmetry']
    data['volume_asymmetry_accel'] = data['volume_asymmetry'] - data['volume_asymmetry'].shift(3)
    
    # Volume Efficiency
    data['volume_efficiency'] = (data['volume'] / (data['high'] - data['low'] + 1e-8)) * data['vol_asymmetry']
    
    # Gap Absorption Dynamics
    # Gap Magnitude
    data['prev_close'] = data['close'].shift(1)
    data['gap_magnitude'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    
    # Absorption Efficiency
    data['absorption_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['vol_asymmetry']
    
    # Regime-Specific Closure
    data['gap_closure'] = (data['close'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    data['regime_closure'] = data['gap_closure'] * data['asymmetry_regime']
    
    # Amount Asymmetry Patterns
    # Direction Consistency
    data['amount_direction'] = np.sign(data['close'] - data['open'])
    data['amount_consistency'] = data['amount'] * data['amount_direction']
    
    # Position Establishment
    data['opening_amount'] = data['amount'] * data['vol_asymmetry']
    
    # Regime Behavior
    data['amount_regime_change'] = data['amount_consistency'] * data['asymmetry_regime'].diff()
    
    # Asymmetry-Divergence Integration
    # Combined Asymmetry
    data['volume_timing'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    data['combined_asymmetry'] = data['vol_asymmetry'] * data['volume_timing']
    
    # Regime Amplification
    data['regime_amplification'] = data['combined_asymmetry'] * data['asymmetry_regime']
    
    # State Alignment
    data['asymmetry_trend_5d'] = data['vol_asymmetry'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    data['state_alignment'] = data['regime_amplification'] * data['asymmetry_trend_5d']
    
    # Alpha Signal Construction
    # Signal Generation
    data['signal'] = data['state_alignment'] * data['asymmetry_trend_5d']
    
    # Intensity Modulation
    data['asymmetry_momentum'] = data['vol_asymmetry'] - data['vol_asymmetry'].shift(3)
    data['intensity'] = data['signal'] * data['volume_timing'] * data['asymmetry_momentum']
    
    # Final Alpha
    data['final_alpha'] = data['intensity'] * (data['regime_amplification'] * data['asymmetry_regime'])
    
    return data['final_alpha']
