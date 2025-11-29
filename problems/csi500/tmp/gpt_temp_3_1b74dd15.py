import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Volatility-Adjusted Momentum (IVAM)
    # Momentum calculation: (Close - Low) / (High - Low)
    momentum = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Persistence signal: 5-day correlation with lagged momentum
    momentum_lag1 = momentum.shift(1)
    ivam_corr = momentum.rolling(window=5, min_periods=3).corr(momentum_lag1)
    ivam_factor = momentum * ivam_corr
    
    # Factor 2: Range-Based Gap Efficiency (RBGE)
    # Gap significance: |Open_t/Close_{t-1} - 1| / (High_t - Low_t)
    gap_magnitude = abs(data['open'] / data['close'].shift(1) - 1)
    daily_range = data['high'] - data['low']
    gap_efficiency = gap_magnitude / daily_range.replace(0, np.nan)
    
    # Gap momentum: 3-day gap sequence analysis
    gap_momentum = gap_efficiency.rolling(window=3, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else np.nan
    )
    rbge_factor = gap_efficiency * gap_momentum
    
    # Factor 3: Volume-Weighted Price Efficiency (VWPE)
    # Efficiency: |Close_t - Close_{t-1}| / (High_t - Low_t) × Volume
    price_change = abs(data['close'] - data['close'].shift(1))
    price_efficiency = price_change / daily_range.replace(0, np.nan)
    volume_weighted_efficiency = price_efficiency * data['volume']
    
    # Momentum: 3-day efficiency trend
    vwpe_momentum = volume_weighted_efficiency.rolling(window=3, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else np.nan
    )
    vwpe_factor = volume_weighted_efficiency * vwpe_momentum
    
    # Factor 4: Amount-Price Volatility Divergence (APVD)
    # Trend consistency: sign(Close_t - Close_{t-1}) × sign(Amount_t - Amount_{t-1})
    price_trend = np.sign(data['close'] - data['close'].shift(1))
    amount_trend = np.sign(data['amount'] - data['amount'].shift(1))
    trend_consistency = price_trend * amount_trend
    
    # Volatility context: 10-day range volatility comparison
    range_volatility = (data['high'] - data['low']).rolling(window=10, min_periods=5).std()
    current_range = data['high'] - data['low']
    volatility_context = current_range / range_volatility.replace(0, np.nan)
    
    apvd_factor = trend_consistency * volatility_context
    
    # Factor 5: Volatility-Regime Reversal Efficiency (VRRE)
    # Regime identification: 10-day range volatility classification
    rolling_range_vol = (data['high'] - data['low']).rolling(window=10, min_periods=5).std()
    regime_threshold = rolling_range_vol.rolling(window=20, min_periods=10).quantile(0.5)
    high_vol_regime = (rolling_range_vol > regime_threshold).astype(int)
    
    # Reversal efficiency: previous return reversal × range utilization
    prev_return = data['close'].pct_change(1)
    range_utilization = abs(data['close'] - data['open']) / daily_range.replace(0, np.nan)
    reversal_efficiency = -prev_return.shift(1) * range_utilization
    
    vrre_factor = high_vol_regime * reversal_efficiency
    
    # Combine factors (equal weighting for simplicity)
    final_factor = (
        ivam_factor.fillna(0) + 
        rbge_factor.fillna(0) + 
        vwpe_factor.fillna(0) + 
        apvd_factor.fillna(0) + 
        vrre_factor.fillna(0)
    )
    
    return final_factor
