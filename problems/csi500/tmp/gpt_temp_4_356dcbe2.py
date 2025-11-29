import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Efficiency-Momentum Convergence Factor combining price efficiency analysis,
    momentum-efficiency alignment, and volume-amount validation.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Path Efficiency Analysis
    # Daily efficiency ratio (Open-to-Close / Sum of intraday fluctuations)
    daily_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # 3-day efficiency trend persistence
    efficiency_3d_trend = daily_efficiency.rolling(window=3).mean()
    efficiency_persistence = daily_efficiency.rolling(window=3).apply(
        lambda x: np.corrcoef(range(3), x)[0,1] if not np.isnan(x).any() else np.nan
    )
    
    # Efficiency breakout detection
    efficiency_breakout = (daily_efficiency > daily_efficiency.rolling(window=5).mean() + 
                          daily_efficiency.rolling(window=5).std())
    
    # Momentum-Efficiency Alignment
    # Price momentum (5-day return)
    momentum_5d = data['close'].pct_change(5)
    
    # High efficiency with strong momentum (bullish convergence)
    bullish_convergence = ((daily_efficiency > daily_efficiency.rolling(window=10).quantile(0.7)) & 
                          (momentum_5d > momentum_5d.rolling(window=10).quantile(0.7)))
    
    # Low efficiency with weak momentum (bearish convergence)
    bearish_convergence = ((daily_efficiency < daily_efficiency.rolling(window=10).quantile(0.3)) & 
                          (momentum_5d < momentum_5d.rolling(window=10).quantile(0.3)))
    
    # Efficiency-momentum divergence patterns
    efficiency_momentum_divergence = (daily_efficiency.rolling(window=5).mean() - 
                                     momentum_5d.rolling(window=5).mean())
    
    # Multi-day convergence persistence
    convergence_persistence = bullish_convergence.rolling(window=3).sum() - bearish_convergence.rolling(window=3).sum()
    
    # Volume-Amount Validation
    # Volume concentration during efficient moves
    volume_concentration = (data['volume'] * np.abs(daily_efficiency)) / (data['volume'].rolling(window=5).mean() + 1e-8)
    
    # Price move efficiency per unit amount
    amount_efficiency = (np.abs(data['close'] - data['open']) / (data['amount'] + 1e-8))
    
    # Smart money flow patterns
    smart_money_flow = ((data['close'] > data['open']) & 
                       (data['volume'] > data['volume'].rolling(window=10).mean()) & 
                       (data['amount'] > data['amount'].rolling(window=10).mean()))
    
    # Combine factors with appropriate weights
    factor = (
        0.25 * daily_efficiency +
        0.15 * efficiency_persistence.fillna(0) +
        0.10 * efficiency_breakout.astype(float) +
        0.20 * bullish_convergence.astype(float) -
        0.20 * bearish_convergence.astype(float) +
        0.15 * efficiency_momentum_divergence.fillna(0) +
        0.10 * convergence_persistence.fillna(0) +
        0.15 * volume_concentration.fillna(0) -
        0.10 * amount_efficiency.fillna(0) +
        0.10 * smart_money_flow.astype(float)
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20).mean()) / (factor.rolling(window=20).std() + 1e-8)
    
    return factor
