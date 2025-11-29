import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Rejection Momentum (VRARM) factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Price Rejection Analysis
    # Rejection strength calculation
    intraday_range = data['high'] - data['low']
    intraday_range = intraday_range.replace(0, np.nan)  # Avoid division by zero
    
    upper_rejection = (data['high'] - data['close']) / intraday_range
    lower_rejection = (data['close'] - data['low']) / intraday_range
    net_rejection = upper_rejection - lower_rejection
    
    # Rejection quality assessment
    rejection_magnitude = np.abs(net_rejection)
    
    # Rejection persistence (5-day rolling consistency)
    rejection_persistence = net_rejection.rolling(window=5).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else np.nan
    )
    
    # 2. Volatility Regime Classification
    # Current volatility assessment
    price_efficiency = np.abs(data['close'] - data['open']) / intraday_range
    price_efficiency = price_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Volatility regime identification (20-day rolling median)
    median_range = intraday_range.rolling(window=20).median()
    high_vol_regime = (intraday_range > median_range).astype(int)
    low_vol_regime = (intraday_range <= median_range).astype(int)
    
    # Regime confidence (price efficiency stability over 5 days)
    regime_confidence = 1 - price_efficiency.rolling(window=5).std()
    
    # 3. Volume-Price Rejection Confirmation
    # Volume concentration analysis
    volume_intensity = data['volume'] / intraday_range
    volume_intensity = volume_intensity.replace([np.inf, -np.inf], np.nan)
    
    volume_persistence = data['volume'] / data['volume'].shift(1)
    volume_persistence = volume_persistence.replace([np.inf, -np.inf], np.nan)
    
    # Rejection-volume synchronization
    high_volume_rejection = net_rejection * volume_intensity
    persistent_rejection = net_rejection * volume_persistence
    volume_confirmed_rejection = high_volume_rejection * persistent_rejection
    
    # 4. Regime-Adaptive Rejection Momentum
    # High volatility dynamics
    sharp_rejection_momentum = volume_confirmed_rejection * price_efficiency
    momentum_collapse = rejection_magnitude * volume_intensity
    
    # Low volatility dynamics
    gradual_rejection = volume_confirmed_rejection / price_efficiency.replace(0, np.nan)
    range_bound_rejection = net_rejection * volume_persistence
    
    # Volatility-weighted rejection signals
    high_vol_signal = sharp_rejection_momentum * (1 - momentum_collapse.rolling(window=3).mean())
    low_vol_signal = gradual_rejection + range_bound_rejection
    
    volatility_weighted_rejection = (
        high_vol_regime * high_vol_signal + 
        low_vol_regime * low_vol_signal
    )
    
    # 5. Multi-timeframe rejection confirmation
    # Short-term rejection alignment (3-day rolling)
    intraday_consistency = net_rejection.rolling(window=3).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 3 else np.nan
    )
    
    # Overnight gap rejection patterns
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_rejection_alignment = np.sign(net_rejection) * np.sign(overnight_gap)
    
    # Final composite rejection score
    regime_weight = high_vol_regime * regime_confidence + low_vol_regime * (1 - regime_confidence)
    
    final_factor = (
        volatility_weighted_rejection * 
        regime_weight * 
        intraday_consistency * 
        (1 + gap_rejection_alignment) *
        rejection_persistence
    )
    
    return final_factor
