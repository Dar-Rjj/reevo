import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['returns'] = data['close'].pct_change()
    data['hl_range'] = (data['high'] - data['low']) / data['close']
    data['oc_range'] = abs(data['close'] - data['open']) / data['close']
    
    # Volatility regime detection
    # Short-term volatility (5-day)
    data['vol_short'] = data['returns'].rolling(window=5).std()
    # Medium-term volatility (20-day)
    data['vol_medium'] = data['returns'].rolling(window=20).std()
    # Volatility ratio for regime detection
    data['vol_ratio'] = data['vol_short'] / data['vol_medium']
    
    # Volatility regime boundaries
    # Detect volatility breakpoints using volatility ratio changes
    data['vol_ratio_change'] = data['vol_ratio'].diff()
    data['vol_break'] = abs(data['vol_ratio_change']) > data['vol_ratio_change'].rolling(window=10).std()
    
    # Volatility regime intensity
    data['regime_intensity'] = abs(data['vol_ratio_change']).rolling(window=5).mean()
    
    # Volatility persistence patterns
    data['vol_persistence'] = data['vol_short'].rolling(window=10).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Microstructure momentum analysis
    # Volume-price momentum divergence
    data['volume_momentum'] = data['volume'].pct_change(periods=3)
    data['price_momentum'] = data['close'].pct_change(periods=3)
    data['momentum_divergence'] = data['volume_momentum'] - data['price_momentum']
    
    # Order flow patterns using amount and volume
    data['tick_size_estimate'] = data['amount'] / data['volume']
    data['tick_momentum'] = data['tick_size_estimate'].pct_change(periods=2)
    
    # Intraday momentum clustering
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['momentum_persistence'] = data['intraday_momentum'].rolling(window=5).apply(
        lambda x: 1 if (x > 0).all() or (x < 0).all() else 0
    )
    
    # Combined regime transition and microstructure signals
    # Transition intensity weighted by momentum strength
    data['transition_momentum'] = data['regime_intensity'] * abs(data['momentum_divergence'])
    
    # Volatility regime complexity
    data['regime_complexity'] = data['vol_break'].rolling(window=10).sum() / 10
    
    # Microstructure momentum decay (exponential weighting)
    decay_weights = np.exp(-np.arange(5) / 2)  # Exponential decay
    data['momentum_decay'] = data['momentum_divergence'].rolling(window=5).apply(
        lambda x: np.average(x, weights=decay_weights[:len(x)]) if len(x) > 0 else 0
    )
    
    # Signal integration across periods
    data['signal_coherence'] = (
        data['intraday_momentum'].rolling(window=3).std() * 
        data['momentum_persistence'].rolling(window=3).mean()
    )
    
    # Nonlinear signal transformation
    # Exponential regime transition effects
    data['exp_transition'] = np.exp(-1 / (1 + abs(data['regime_intensity'])))
    
    # Hyperbolic decay for microstructure momentum
    data['hyperbolic_momentum'] = data['momentum_decay'] / (1 + abs(data['momentum_decay']))
    
    # Final factor calculation combining all components
    factor = (
        data['transition_momentum'] * 
        data['exp_transition'] * 
        data['hyperbolic_momentum'] * 
        (1 + data['regime_complexity']) * 
        data['signal_coherence']
    )
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
