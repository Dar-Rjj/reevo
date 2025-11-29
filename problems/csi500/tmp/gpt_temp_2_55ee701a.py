import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Volatility-Efficiency Alpha Factor
    Combines range compression, session transitions, price-volume alignment, 
    range extension, and gap-fill efficiency signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for volatility calculation
    def calc_volatility(series, window=20):
        return series.rolling(window=window).std()
    
    # 1. Range Compression Efficiency
    # Multi-Timeframe Range Analysis
    daily_range = (data['high'] - data['low']) / data['close']
    range_5d = daily_range.rolling(window=5).mean()
    range_20d = daily_range.rolling(window=20).mean()
    range_compression = (range_5d - range_20d) / range_20d
    
    # Volatility-Regime Efficiency Anomalies
    vol_20d = calc_volatility(data['close'], 20)
    range_efficiency = daily_range / vol_20d
    range_efficiency_signal = (range_efficiency - range_efficiency.rolling(window=20).mean()) / range_efficiency.rolling(window=20).std()
    
    # 2. Session Transition Momentum
    # Pre-Close to Post-Open Dynamics
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    intraday_return = (data['close'] - data['open']) / data['open']
    
    # Volatility-Adjusted Transition Patterns
    gap_vol_adj = overnight_gap / vol_20d
    intraday_vol_adj = intraday_return / vol_20d
    transition_momentum = gap_vol_adj * intraday_vol_adj
    
    # 3. Price-Volume Fractal Alignment
    # Volume-Weighted Price Efficiency
    vwap = (data['close'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    price_vwap_ratio = data['close'] / vwap
    volume_efficiency = data['volume'] / data['volume'].rolling(window=20).mean()
    price_volume_alignment = price_vwap_ratio * volume_efficiency
    
    # Multi-Timeframe Fractal Coherence
    short_ma = data['close'].rolling(window=5).mean()
    medium_ma = data['close'].rolling(window=20).mean()
    fractal_coherence = (short_ma / medium_ma - 1) * volume_efficiency
    
    # 4. Range Extension Compression
    # Tail Range Utilization
    upper_tail_util = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    lower_tail_util = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    tail_asymmetry = upper_tail_util - lower_tail_util
    
    # Volatility-Weighted Compression Signals
    range_utilization = daily_range / daily_range.rolling(window=20).mean()
    compression_signal = tail_asymmetry * range_utilization
    
    # 5. Gap-Fill Efficiency
    # Gap Size vs Volatility Context
    gap_magnitude = abs(overnight_gap)
    gap_vol_ratio = gap_magnitude / vol_20d
    
    # Volatility-Weighted Fill Momentum
    gap_fill_ratio = abs(intraday_return) / gap_magnitude.replace(0, np.nan)
    fill_efficiency = gap_fill_ratio * gap_vol_ratio
    
    # Combine all components with equal weighting
    factor = (
        0.2 * range_compression +
        0.2 * range_efficiency_signal +
        0.15 * transition_momentum +
        0.15 * price_volume_alignment +
        0.1 * fractal_coherence +
        0.1 * compression_signal +
        0.1 * fill_efficiency
    )
    
    # Cross-sectional normalization
    def cross_sectional_rank(series):
        return series.groupby(series.index).rank(pct=True) - 0.5
    
    # Apply cross-sectional ranking to final factor
    factor_cs = factor.groupby(factor.index).apply(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return factor_cs
