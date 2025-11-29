import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility Regime Components
    # Intraday Volatility Metrics
    data['hl_range'] = (data['high'] - data['low']) / data['close']
    data['oc_gap'] = abs(data['close'] - data['open']) / data['open']
    data['price_oscillation'] = (data['high'] - data['low']) / (data['high'] + data['low']) * 2
    
    # Volatility Regime Indicators
    data['vol_clustering'] = data['hl_range'].rolling(window=5).std() / data['hl_range'].rolling(window=20).std()
    data['vol_persistence'] = data['hl_range'].rolling(window=10).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else np.nan)
    data['vol_transition'] = data['hl_range'].pct_change().abs()
    
    # Liquidity Dynamics
    # Liquidity Flow Metrics
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['amount_liquidity'] = data['amount'] / data['close']
    data['price_impact'] = (data['high'] - data['low']) / data['amount'] * 1e6
    
    # Regime Shift Detection
    data['vol_liq_correlation'] = data['hl_range'].rolling(window=10).corr(data['volume_concentration'])
    data['liquidity_evaporation'] = (data['volume_concentration'].rolling(window=5).mean() - 
                                   data['volume_concentration'].rolling(window=20).mean())
    data['market_depth'] = data['amount'] / data['hl_range']
    
    # Regime Transition Signals
    # Volatility Breakouts
    data['vol_breakout'] = (data['hl_range'] - data['hl_range'].rolling(window=20).mean()) / data['hl_range'].rolling(window=20).std()
    data['vol_clustering_intensity'] = data['vol_clustering'].rolling(window=5).mean()
    data['regime_persistence'] = data['vol_breakout'].rolling(window=10).apply(lambda x: len([i for i in range(1, len(x)) if x[i] * x[i-1] > 0]))
    
    # Liquidity Transitions
    data['liquidity_acceleration'] = data['volume_concentration'].pct_change(3)
    data['volume_amount_divergence'] = (data['volume_concentration'] - data['amount_liquidity'].rolling(window=5).mean()) / data['amount_liquidity'].rolling(window=5).std()
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Generate Final Regime Factors
    # Volatility regime strength (normalized combination of volatility signals)
    vol_strength = (data['vol_breakout'].fillna(0) + 
                   data['vol_clustering_intensity'].fillna(0) + 
                   data['vol_persistence'].fillna(0))
    data['volatility_regime_strength'] = (vol_strength - vol_strength.rolling(window=20).mean()) / vol_strength.rolling(window=20).std()
    
    # Liquidity regime quality (combination of liquidity signals)
    liq_quality = (data['liquidity_evaporation'].fillna(0) + 
                  data['market_depth'].fillna(0) - 
                  data['price_impact'].fillna(0))
    data['liquidity_regime_quality'] = (liq_quality - liq_quality.rolling(window=20).mean()) / liq_quality.rolling(window=20).std()
    
    # Regime transition timing (detection of regime changes)
    regime_change = (data['vol_transition'].fillna(0) * data['liquidity_acceleration'].fillna(0) * 
                    data['volume_amount_divergence'].fillna(0))
    data['regime_transition_timing'] = (regime_change - regime_change.rolling(window=20).mean()) / regime_change.rolling(window=20).std()
    
    # Final factor combining all regime components
    final_factor = (data['volatility_regime_strength'].fillna(0) + 
                   data['liquidity_regime_quality'].fillna(0) + 
                   data['regime_transition_timing'].fillna(0))
    
    # Normalize the final factor
    final_factor = (final_factor - final_factor.rolling(window=20).mean()) / final_factor.rolling(window=20).std()
    
    return final_factor
