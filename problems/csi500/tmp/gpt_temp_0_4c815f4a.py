import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Compression Measurement
    # Daily Compression Ratio
    daily_compression_ratio = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Opening Gap Compression
    opening_gap_compression = abs(data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Intraday Compression Score
    intraday_compression_score = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volume Expansion Patterns
    # Volume Breakout Signal
    volume_breakout_signal = (data['volume'] / data['volume'].shift(1)) * np.log(data['volume'] + 1)
    
    # Volume Persistence
    volume_persistence = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Volume-Price Expansion
    volume_price_expansion = volume_breakout_signal * abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Multi-period Momentum Structure
    # Short-term Momentum
    short_term_momentum = (data['close'] - data['close'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2))
    
    # Medium-term Momentum
    medium_term_momentum = (data['close'] - data['close'].shift(5)) / (data['high'].shift(5) - data['low'].shift(5))
    
    # Momentum Convergence
    momentum_convergence = short_term_momentum / medium_term_momentum
    
    # Volatility Regime Detection
    # Calculate price returns
    returns = data['close'] / data['close'].shift(1) - 1
    
    # Recent Volatility (3-day std of 5-day returns)
    recent_volatility = returns.rolling(window=5).apply(lambda x: x.pct_change().std(), raw=False).rolling(window=3).std()
    
    # Volatility Regime (ratio of recent vs longer-term volatility)
    volatility_regime = recent_volatility / returns.rolling(window=10).apply(lambda x: x.pct_change().std(), raw=False).rolling(window=5).std()
    
    # Final Alpha Construction
    # Compression-Expansion Core
    compression_expansion_core = daily_compression_ratio * volume_price_expansion
    
    # Momentum Confirmation
    momentum_confirmation = compression_expansion_core * momentum_convergence
    
    # Regime-Weighted Alpha
    regime_weight = np.where(volatility_regime > 1, 2.0, 0.5)
    final_alpha = momentum_confirmation * regime_weight
    
    return final_alpha
