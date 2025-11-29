import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Volatility-Efficiency Analysis
    # Morning Volatility Proxy: (First Hour High - First Hour Low) / Open
    # Since we only have daily OHLC, we'll use the day's range as proxy
    data['morning_vol'] = (data['high'] - data['low']) / data['open']
    
    # Afternoon Volatility Proxy: (Last Hour High - Last Hour Low) / Midday Price
    # Using midday price as (high + low)/2
    midday_price = (data['high'] + data['low']) / 2
    data['afternoon_vol'] = (data['high'] - data['low']) / midday_price
    
    # True Range calculation
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    data['true_range'] = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Efficiency Ratio: Amount / (Volume × True Range)
    data['efficiency_ratio'] = data['amount'] / (data['volume'] * data['true_range'])
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # 2. Volatility Regime Classification with Efficiency
    # Volatility Ratio: ln(Morning Volatility / Afternoon Volatility)
    vol_ratio = np.log(data['morning_vol'] / data['afternoon_vol'])
    
    # Classify regimes using rolling percentiles
    vol_ratio_roll = vol_ratio.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Regime classification
    data['vol_regime'] = 0  # Neutral
    data.loc[vol_ratio_roll > 0.7, 'vol_regime'] = 1  # Compression
    data.loc[vol_ratio_roll < 0.3, 'vol_regime'] = -1  # Expansion
    
    # 3. Volume-Weighted Breakout Detection
    # Range Expansion Analysis
    prev_range = (data['high'].shift(1) - data['low'].shift(1))
    current_range = data['high'] - data['low']
    data['range_expansion'] = current_range / prev_range - 1
    
    # Volume-Weighted Intraday Momentum
    intraday_return = (data['close'] - data['open']) / data['open']
    data['volume_weighted_signal'] = data['range_expansion'] * intraday_return * data['volume']
    
    # 4. Extreme Price Reversal with Volatility Confirmation
    # Local Extremum Detection using rolling windows
    data['local_high'] = data['high'].rolling(window=3, center=False).max()
    data['local_low'] = data['low'].rolling(window=3, center=False).min()
    
    # Price Reversal
    price_reversal = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Volatility Adjustment
    data['vol_adjusted_reversal'] = price_reversal * data['morning_vol']
    
    # 5. Gap Momentum with Volatility-Efficiency Filter
    # Opening Gap Analysis
    gap_size = data['open'] - data['close'].shift(1)
    gap_return = gap_size / data['close'].shift(1)
    
    # Volatility-Efficiency Weighting
    gap_signal = gap_return * data['efficiency_ratio']
    
    # Apply Volatility Regime Filter
    gap_signal_weighted = gap_signal.copy()
    gap_signal_weighted[data['vol_regime'] == 1] *= 1.5   # Compression: amplify
    gap_signal_weighted[data['vol_regime'] == -1] *= 0.5  # Expansion: dampen
    
    # 6. Multi-Dimensional Signal Integration
    # Core Volatility-Efficiency Momentum
    core_momentum = data['volume_weighted_signal'] * data['efficiency_ratio']
    
    # Apply Volatility Regime Modulation
    core_momentum_weighted = core_momentum.copy()
    core_momentum_weighted[data['vol_regime'] == 1] *= 1.2   # Compression
    core_momentum_weighted[data['vol_regime'] == -1] *= 0.8  # Expansion
    
    # Enhanced Reversal Component
    enhanced_reversal = core_momentum_weighted * data['vol_adjusted_reversal']
    final_signal = enhanced_reversal * gap_signal_weighted
    
    # Regime-Adaptive Signal Processing
    alpha_signal = final_signal.copy()
    
    # Compression regime: focus on continuation signals
    compression_mask = data['vol_regime'] == 1
    alpha_signal[compression_mask] = alpha_signal[compression_mask] * np.sign(core_momentum_weighted[compression_mask])
    
    # Expansion regime: emphasize reversal signals  
    expansion_mask = data['vol_regime'] == -1
    alpha_signal[expansion_mask] = alpha_signal[expansion_mask] * -np.sign(data['vol_adjusted_reversal'][expansion_mask])
    
    # Neutral regime: balanced signal combination (no additional processing)
    
    # Final alpha generation with normalization
    alpha = alpha_signal.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    return alpha
