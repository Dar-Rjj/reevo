import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining fractal momentum dynamics, 
    liquidity wave propagation, and price gap echo analysis
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Scale Fractal Momentum Divergence
    # Fractal Time Series Decomposition
    data['hl_range'] = data['high'] - data['low']
    data['price_path_3d'] = data['close'].rolling(window=3).apply(
        lambda x: np.sum(np.abs(np.diff(x))) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0, 
        raw=True
    )
    data['price_path_10d'] = data['close'].rolling(window=10).apply(
        lambda x: np.sum(np.abs(np.diff(x))) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0, 
        raw=True
    )
    data['fractal_complexity'] = data['price_path_3d'] - data['price_path_10d']
    
    # High-Low range fractal dimension
    data['hl_fractal'] = np.log(data['hl_range'].rolling(window=5).std() + 1e-6) / np.log(5)
    
    # Volume clustering pattern
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['volume_clustering'] = data['volume_zscore'].rolling(window=5).apply(
        lambda x: np.sum(np.abs(x) > 1) / len(x), raw=True
    )
    
    # Momentum Divergence Across Scales
    data['momentum_1d'] = data['close'].pct_change(1)
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_divergence'] = np.sign(data['momentum_1d']) * np.sign(data['momentum_5d'])
    
    data['volume_momentum'] = data['volume'].pct_change(3)
    data['price_volume_divergence'] = data['momentum_1d'] - data['volume_momentum']
    
    # Adaptive Fractal Signal
    data['fractal_momentum_signal'] = (
        data['fractal_complexity'] * data['momentum_divergence'] * 
        (1 + data['volume_clustering'])
    )
    
    # 2. Liquidity Wave Propagation Analysis
    # Multi-Asset Liquidity Spillover (using rolling correlation with market volume)
    market_volume = data['volume'].groupby(level=0).transform('mean')
    data['volume_correlation'] = data['volume'].rolling(window=10).corr(market_volume)
    
    # Price impact asymmetry
    data['price_impact'] = (data['high'] - data['low']) / (data['volume'] + 1e-6)
    data['price_impact_momentum'] = data['price_impact'].pct_change(3)
    
    # Liquidity Momentum Construction
    data['volume_acceleration'] = data['volume'].pct_change(3) - data['volume'].pct_change(1)
    data['bid_ask_proxy'] = (data['high'] - data['low']) / data['close']
    data['bid_ask_momentum'] = data['bid_ask_proxy'].pct_change(3)
    
    # Liquidity clustering persistence
    data['liquidity_clustering'] = data['volume'].rolling(window=5).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-6), raw=True
    )
    
    # Wave-Based Signal
    data['liquidity_wave_signal'] = (
        data['volume_correlation'] * data['volume_acceleration'] * 
        data['bid_ask_momentum'] * (1 - data['liquidity_clustering'])
    )
    
    # 3. Price Gap Echo Analysis
    # Multi-Period Gap Resonance
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence_3d'] = data['overnight_gap'].rolling(window=3).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 3 else 0, 
        raw=False
    )
    
    # Intraday gap absorption
    data['intraday_gap'] = (data['close'] - data['open']) / data['open']
    data['gap_absorption'] = np.abs(data['intraday_gap']) / (np.abs(data['overnight_gap']) + 1e-6)
    
    # Gap magnitude decay patterns
    data['gap_decay'] = data['overnight_gap'].rolling(window=3).apply(
        lambda x: np.abs(x.iloc[-1]) / (np.mean(np.abs(x)) + 1e-6) if len(x) == 3 else 0,
        raw=False
    )
    
    # Cross-Sectional Gap Analysis (using rolling z-score for cross-sectional comparison)
    data['gap_zscore'] = data.groupby(level=0)['overnight_gap'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    
    # Echo-Based Signal
    data['gap_echo_signal'] = (
        data['gap_persistence_3d'] * data['gap_absorption'] * 
        data['gap_decay'] * data['gap_zscore']
    )
    
    # 4. Dynamic Multi-Factor Synthesis
    # Fractal Regime Detection
    data['volatility_5d'] = data['close'].pct_change().rolling(window=5).std()
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['volatility_regime'] = data['volatility_5d'] / (data['volatility_20d'] + 1e-6)
    
    # Fractal dimension regime
    data['fractal_regime'] = data['hl_fractal'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-6), raw=False
    )
    
    # Regime classification
    data['high_vol_regime'] = (data['volatility_regime'] > 1.2).astype(int)
    data['high_fractal_regime'] = (data['fractal_regime'] > 0).astype(int)
    
    # Adaptive Factor Weighting
    # High volatility regime weights
    high_vol_weights = {
        'fractal': 0.4,
        'liquidity': 0.3,
        'gap': 0.3
    }
    
    # Low volatility regime weights  
    low_vol_weights = {
        'fractal': 0.3,
        'liquidity': 0.4,
        'gap': 0.3
    }
    
    # High fractal regime weights
    high_fractal_weights = {
        'fractal': 0.5,
        'liquidity': 0.25,
        'gap': 0.25
    }
    
    # Calculate regime-adaptive signals
    data['fractal_component'] = data['fractal_momentum_signal'].fillna(0)
    data['liquidity_component'] = data['liquidity_wave_signal'].fillna(0)
    data['gap_component'] = data['gap_echo_signal'].fillna(0)
    
    # Apply regime-based weighting
    conditions = [
        (data['high_vol_regime'] == 1) & (data['high_fractal_regime'] == 1),
        (data['high_vol_regime'] == 1) & (data['high_fractal_regime'] == 0),
        (data['high_vol_regime'] == 0) & (data['high_fractal_regime'] == 1),
        (data['high_vol_regime'] == 0) & (data['high_fractal_regime'] == 0)
    ]
    
    choices = [
        high_vol_weights['fractal'] * data['fractal_component'] + 
        high_vol_weights['liquidity'] * data['liquidity_component'] + 
        high_vol_weights['gap'] * data['gap_component'],
        
        high_vol_weights['fractal'] * data['fractal_component'] + 
        high_vol_weights['liquidity'] * data['liquidity_component'] + 
        high_vol_weights['gap'] * data['gap_component'],
        
        high_fractal_weights['fractal'] * data['fractal_component'] + 
        high_fractal_weights['liquidity'] * data['liquidity_component'] + 
        high_fractal_weights['gap'] * data['gap_component'],
        
        low_vol_weights['fractal'] * data['fractal_component'] + 
        low_vol_weights['liquidity'] * data['liquidity_component'] + 
        low_vol_weights['gap'] * data['gap_component']
    ]
    
    data['alpha_raw'] = np.select(conditions, choices, default=0)
    
    # Final cross-sectional normalization
    alpha_final = data.groupby(level=0)['alpha_raw'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    
    return alpha_final
