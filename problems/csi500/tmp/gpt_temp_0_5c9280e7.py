import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Momentum Structure alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume features
    df['returns'] = df['close'].pct_change()
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['oc_gap'] = (df['close'] - df['open']) / df['open']
    df['volume_norm'] = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-timeframe Price Fractal Dimension
        # 3-day vs 8-day High-Low range fractal scaling
        hl_3d = current_data['hl_range'].tail(3).std()
        hl_8d = current_data['hl_range'].tail(8).std()
        price_fractal_3v8 = hl_3d / (hl_8d + 1e-8) if hl_8d > 0 else 0
        
        # Close price fractal structure across 5-day windows
        close_5d = current_data['close'].tail(5)
        close_fractal = (close_5d.max() - close_5d.min()) / (close_5d.std() + 1e-8)
        
        # Open-Close gap fractal patterns
        oc_5d = current_data['oc_gap'].tail(5).abs()
        oc_fractal = oc_5d.std() / (oc_5d.mean() + 1e-8)
        
        # Price Momentum Self-Similarity
        # Short-term momentum persistence (1d vs 3d)
        mom_1d = current_data['returns'].tail(1).iloc[0] if len(current_data) > 0 else 0
        mom_3d = current_data['returns'].tail(3).mean()
        mom_persistence = np.sign(mom_1d) * np.sign(mom_3d) if mom_3d != 0 else 0
        
        # Medium-term momentum fractal replication (3d vs 5d)
        mom_5d = current_data['returns'].tail(5).mean()
        mom_replication = 1 - abs(mom_3d - mom_5d) / (abs(mom_3d) + abs(mom_5d) + 1e-8)
        
        # Volume Fractal Dynamics
        # Volume distribution fractal dimension
        vol_5d = current_data['volume_norm'].tail(5)
        vol_fractal = vol_5d.std() / (vol_5d.mean() + 1e-8)
        
        # Volume clustering patterns
        vol_cluster = (vol_5d > vol_5d.median()).sum() / 5.0
        
        # Price-Volume Fractal Coupling
        # Correlation between price and volume fractal patterns
        price_vol_corr = np.corrcoef(current_data['returns'].tail(5).fillna(0), 
                                   current_data['volume_norm'].tail(5).fillna(0))[0,1]
        price_vol_corr = 0 if np.isnan(price_vol_corr) else price_vol_corr
        
        # Fractal Regime Change Detection
        # Price fractal dimension regime shifts
        price_fractal_prev = current_data['hl_range'].iloc[i-5:i].std() if i >= 5 else 0
        price_fractal_curr = current_data['hl_range'].tail(5).std()
        fractal_regime_shift = (price_fractal_curr - price_fractal_prev) / (price_fractal_prev + 1e-8)
        
        # Multi-Scale Momentum Alignment
        # Short-medium-long term momentum coherence
        mom_short = current_data['returns'].tail(3).mean()
        mom_medium = current_data['returns'].tail(8).mean()
        mom_long = current_data['returns'].tail(13).mean()
        
        momentum_coherence = (np.sign(mom_short) * np.sign(mom_medium) + 
                            np.sign(mom_medium) * np.sign(mom_long)) / 2.0
        
        # Fractal Pattern Strength Analysis
        # Strong fractal momentum structure components
        fractal_strength = (price_fractal_3v8 + close_fractal + oc_fractal) / 3.0
        
        # Price-volume fractal coupling strength
        coupling_strength = abs(price_vol_corr) * vol_fractal
        
        # Cross-scale momentum fragmentation
        momentum_fragmentation = (abs(mom_short - mom_medium) + abs(mom_medium - mom_long)) / 2.0
        
        # Generate Fractal Alpha Factors
        
        # Fractal Momentum Coherence Factor
        coherence_factor = (fractal_strength * (1 + coupling_strength) * 
                          (1 + momentum_coherence) * mom_persistence * mom_replication)
        
        # Fractal Transition Signal
        transition_signal = (-fractal_regime_shift * momentum_fragmentation * 
                           (1 - abs(price_vol_corr)))
        
        # Combined alpha factor
        alpha_value = coherence_factor + transition_signal
        
        result.iloc[i] = alpha_value
    
    # Normalize the factor
    result = (result - result.rolling(window=20, min_periods=10).mean()) / (result.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return result
