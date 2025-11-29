import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    
    # Gap-Fractal Components
    data['opening_gap_intensity'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['fractal_absorption'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Divergence Detection
    data['gap_momentum_conflict'] = (np.sign(data['opening_gap_intensity']) != np.sign(data['intraday_momentum'])).astype(int)
    data['divergence_strength'] = np.abs(data['opening_gap_intensity']) - np.abs(data['intraday_momentum'])
    
    # Volume Absorption Analysis
    data['absorption_efficiency'] = data['volume'] / np.abs(data['open'] - data['prev_close']).replace(0, np.nan)
    
    # Absorption Confirmation: High volume with gap-momentum divergence
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=10).rank(pct=True)
    data['absorption_confirmation'] = (data['gap_momentum_conflict'] == 1) & (data['volume_rank'] > 0.7)
    
    # Momentum Persistence - Track consecutive divergence patterns
    data['divergence_sequence'] = data['gap_momentum_conflict'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift(1)).cumsum()).cumcount() + 1)
    )
    
    # Volatility-regime weighting
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    data['volatility_regime'] = data['volatility_20d'].rolling(window=60, min_periods=30).rank(pct=True)
    
    # Alpha Signal Generation
    # Base divergence signal
    divergence_signal = data['divergence_strength'] * data['gap_momentum_conflict']
    
    # Enhanced with fractal absorption
    absorption_weighted = divergence_signal * data['fractal_absorption'].fillna(0)
    
    # Persistence enhancement
    persistence_enhanced = absorption_weighted * (1 + 0.1 * data['divergence_sequence'])
    
    # Absorption confirmation boost
    absorption_boosted = np.where(
        data['absorption_confirmation'],
        persistence_enhanced * (1 + 0.2 * data['absorption_efficiency'].fillna(0)),
        persistence_enhanced
    )
    
    # Volatility-regime weighting
    volatility_adjusted = absorption_boosted * (1 - 0.3 * data['volatility_regime'].fillna(0.5))
    
    # Final alpha factor
    alpha_factor = volatility_adjusted
    
    # Clean up and return
    result = pd.Series(alpha_factor, index=data.index)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
