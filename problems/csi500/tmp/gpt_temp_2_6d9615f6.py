import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel cross-sectional alpha factors using intraday price patterns,
    opening momentum, volume decay, price compression, and session transitions.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Price Path Fractality Factor
    # Multi-scale fractal dimension analysis
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['open_close_range'] = abs(data['close'] - data['open']) / data['open']
    
    # Fractal complexity measurement
    data['path_complexity'] = (data['intraday_range'] - data['open_close_range']) / (data['intraday_range'] + 1e-8)
    
    # 3-day fractal pattern evolution
    data['fractal_3d_consistency'] = data['path_complexity'].rolling(window=3).std()
    data['fractal_persistence'] = data['path_complexity'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0, raw=False
    )
    
    # Volatility-adjusted fractal scoring
    vol_20d = data['close'].pct_change().rolling(window=20).std()
    data['fractal_vol_adjusted'] = data['path_complexity'] / (vol_20d + 1e-8)
    
    # Factor 2: Opening Auction Imbalance Momentum Factor
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['first_30min_range'] = (data['high'] - data['low']) / data['open']  # Proxy for first 30min
    
    # Opening momentum persistence
    data['gap_persistence'] = data['overnight_gap'].rolling(window=3).apply(
        lambda x: 1 if all(x > 0) or all(x < 0) else 0, raw=False
    )
    
    # Opening-closing momentum alignment
    data['open_close_alignment'] = np.sign(data['overnight_gap']) * np.sign(data['close'].pct_change())
    data['morning_strength'] = (data['close'] - data['open']) / data['open'] - data['overnight_gap']
    
    # Factor 3: Price-Volume Temporal Decay Factor
    # Volume concentration timing (proxy using daily patterns)
    data['volume_decay'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Price impact decay dynamics
    price_change = data['close'].pct_change()
    data['price_impact'] = abs(price_change) / (data['volume'] + 1e-8)
    data['impact_decay'] = data['price_impact'] / data['price_impact'].rolling(window=5).mean()
    
    # Multi-day decay patterns
    data['volume_decay_3d'] = data['volume_decay'].rolling(window=3).mean()
    data['decay_momentum'] = data['volume_decay'].diff(3)
    
    # Factor 4: Extreme Price Compression Breakout Factor
    # Price range compression
    data['daily_range'] = (data['high'] - data['low']) / data['open']
    data['range_compression'] = data['daily_range'] / data['daily_range'].rolling(window=10).mean()
    
    # 3-day consecutive range narrowing
    data['range_narrowing'] = data['daily_range'].rolling(window=3).apply(
        lambda x: 1 if x.iloc[0] > x.iloc[1] > x.iloc[2] else 0, raw=False
    )
    
    # Volatility squeeze (Bollinger Band width proxy)
    bb_upper = data['close'].rolling(window=20).mean() + 2 * data['close'].rolling(window=20).std()
    bb_lower = data['close'].rolling(window=20).mean() - 2 * data['close'].rolling(window=20).std()
    data['bb_width'] = (bb_upper - bb_lower) / data['close'].rolling(window=20).mean()
    data['vol_squeeze'] = data['bb_width'] / data['bb_width'].rolling(window=20).mean()
    
    # Breakout direction prediction
    data['compression_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Factor 5: Session Boundary Transition Momentum
    # Opening session momentum transfer
    data['gap_fill'] = abs(data['overnight_gap']) - abs((data['close'] - data['open']) / data['open'])
    
    # Intraday session momentum shifts (proxy using afternoon vs morning)
    data['afternoon_momentum'] = (data['close'] - (data['high'] + data['low']) / 2) / data['open']
    
    # Multi-day session pattern recognition
    data['boundary_consistency'] = data['overnight_gap'].rolling(window=3).std()
    data['closing_carryover'] = data['close'].pct_change().rolling(window=3).mean()
    
    # Combine factors with appropriate weights
    factors = pd.DataFrame(index=data.index)
    
    # Fractality Factor (negative weight - complex patterns may mean reversal)
    factors['fractality'] = -0.3 * data['fractal_vol_adjusted'] + 0.2 * data['fractal_persistence']
    
    # Opening Momentum (positive weight - momentum tends to persist)
    factors['opening_momentum'] = 0.4 * data['gap_persistence'] + 0.3 * data['morning_strength']
    
    # Volume Decay (negative weight - decaying volume suggests weakness)
    factors['volume_decay_factor'] = -0.25 * data['decay_momentum'] - 0.2 * data['impact_decay']
    
    # Compression Breakout (positive weight - compression precedes breakouts)
    factors['compression'] = 0.35 * (1 / data['range_compression']) + 0.25 * data['range_narrowing']
    
    # Session Transition (mixed signals)
    factors['session_transition'] = 0.2 * data['gap_fill'] + 0.15 * data['afternoon_momentum']
    
    # Final combined factor
    final_factor = (
        factors['fractality'].fillna(0) +
        factors['opening_momentum'].fillna(0) +
        factors['volume_decay_factor'].fillna(0) +
        factors['compression'].fillna(0) +
        factors['session_transition'].fillna(0)
    )
    
    # Normalize the final factor
    final_factor = (final_factor - final_factor.rolling(window=20).mean()) / (final_factor.rolling(window=20).std() + 1e-8)
    
    return final_factor
