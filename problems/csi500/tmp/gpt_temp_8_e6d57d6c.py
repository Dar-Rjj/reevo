import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Regime Momentum Efficiency Architecture factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Multi-Session Efficiency Quality
    df['session_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_3s_avg'] = df['session_efficiency'].rolling(window=3, min_periods=1).mean()
    df['efficiency_momentum'] = df['session_efficiency'] - df['efficiency_3s_avg']
    df['efficiency_pressure'] = df['session_efficiency'] * df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volatility-Regime Momentum Structure
    df['morning_volatility'] = (df['high'] - df['low']) / df['open'].replace(0, np.nan)
    df['volatility_regime'] = df['morning_volatility'].rolling(window=5, min_periods=1).apply(
        lambda x: 2 if x.iloc[-1] > x.quantile(0.8) else (1 if x.iloc[-1] > x.quantile(0.6) else 0), raw=False
    )
    
    # Anchoring-Flow Momentum Efficiency
    df['anchor_price'] = (df['open'] + df['close']) / 2
    df['anchoring_strength'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Transaction-Pressure Momentum Quality
    df['transaction_efficiency'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['directional_pressure'] = (df['close'] - df['open']) * df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    
    # Gap Elasticity with Momentum Confirmation
    df['prev_close'] = df['close'].shift(1)
    df['opening_gap'] = (df['open'] - df['prev_close']) / df['prev_close'].replace(0, np.nan)
    df['gap_resilience'] = (df['close'] - df['open']) / df['opening_gap'].abs().replace(0, np.nan)
    
    # Volume-Concentrated Efficiency Momentum
    df['volume_concentration'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Multi-Scale Range Expansion Quality
    df['range_expansion'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
    
    # Temporal Momentum Divergence Analysis
    df['efficiency_divergence'] = df['session_efficiency'] - df['session_efficiency'].rolling(window=3, min_periods=1).mean()
    
    # Micro-Trend Quality with Absorption
    df['trend_consistency'] = df['session_efficiency'].rolling(window=3, min_periods=1).std()
    df['absorption_intensity'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Integrated Cross-Regime Framework
    # Momentum Quality Score
    df['momentum_quality'] = df['efficiency_momentum'] * df['directional_pressure']
    
    # Anchoring-Flow Multiplier
    df['anchoring_flow'] = df['anchoring_strength'] * df['absorption_intensity']
    
    # Regime Adjustment
    regime_weights = {
        0: 0.3,  # Low volatility - focus on pressure and divergence
        1: 0.6,  # Medium volatility - balanced approach
        2: 0.8   # High volatility - emphasize efficiency and elasticity
    }
    df['regime_weight'] = df['volatility_regime'].map(regime_weights)
    
    # Final Composite Factor
    df['composite_factor'] = (
        df['momentum_quality'] * 
        df['anchoring_flow'] * 
        df['regime_weight'] * 
        df['gap_resilience'].fillna(0) * 
        (1 - df['trend_consistency'])
    )
    
    # Clean up intermediate columns
    result = df['composite_factor'].copy()
    
    return result
