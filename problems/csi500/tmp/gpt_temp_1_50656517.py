import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate daily returns and ranges
    data['prev_close'] = data['close'].shift(1)
    data['daily_return'] = (data['close'] - data['prev_close']) / data['prev_close']
    data['daily_range'] = data['high'] - data['low']
    data['prev_range'] = data['daily_range'].shift(1)
    
    # 1. Intraday Return Asymmetry
    data['morning_strength'] = (data['high'] - data['open']) / data['open']
    data['afternoon_weakness'] = (data['close'] - data['low']) / data['low']
    data['asymmetry_ratio'] = data['morning_strength'] / data['afternoon_weakness']
    data['asymmetry_ratio'] = data['asymmetry_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # 2. Volume-Price Divergence
    data['up_volume'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['down_volume'] = np.where(data['close'] < data['open'], data['volume'], 0)
    
    # Rolling volume intensities (5-day window)
    data['up_volume_intensity'] = data['up_volume'].rolling(window=5, min_periods=3).mean()
    data['down_volume_intensity'] = data['down_volume'].rolling(window=5, min_periods=3).mean()
    data['volume_polarity'] = data['up_volume_intensity'] / data['down_volume_intensity']
    data['volume_polarity'] = data['volume_polarity'].replace([np.inf, -np.inf], np.nan)
    
    # 3. Gap Reversal Dynamics
    data['overnight_gap'] = abs(data['open'] - data['prev_close']) / data['prev_close']
    data['gap_reversal_strength'] = abs(data['close'] - data['open']) / data['overnight_gap']
    data['gap_reversal_strength'] = data['gap_reversal_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Gap reversal indicator (1 if close moves opposite to gap direction)
    gap_direction = np.sign(data['open'] - data['prev_close'])
    close_direction = np.sign(data['close'] - data['open'])
    data['gap_reversal'] = (gap_direction * close_direction) < 0
    
    # Rolling reversal consistency (10-day window)
    data['reversal_consistency'] = data['gap_reversal'].rolling(window=10, min_periods=5).mean()
    
    # 4. Range Expansion Patterns
    data['range_expansion'] = data['daily_range'] / data['prev_range']
    data['range_expansion'] = data['range_expansion'].replace([np.inf, -np.inf], np.nan)
    
    data['directional_expansion'] = (data['close'] - data['open']) / data['daily_range']
    data['directional_expansion'] = data['directional_expansion'].replace([np.inf, -np.inf], np.nan)
    
    # Expansion persistence (consecutive days with range expansion > 1)
    expansion_flag = data['range_expansion'] > 1
    data['expansion_persistence'] = expansion_flag.rolling(window=5, min_periods=1).apply(
        lambda x: x[::-1].cumprod()[::-1].sum(), raw=False
    )
    
    # 5. Momentum Regime Integration
    # Short-term momentum (5-day return)
    data['momentum_5d'] = data['close'].pct_change(5)
    
    # Asymmetry-momentum alignment
    data['asymmetry_momentum_alignment'] = data['asymmetry_ratio'] * data['momentum_5d']
    
    # Volume-confirmed momentum
    data['volume_confirmed_momentum'] = data['momentum_5d'] * data['volume_polarity']
    
    # Cross-sectional normalization function
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    # Calculate cross-sectional scores for each component
    components = [
        'asymmetry_ratio',
        'volume_polarity', 
        'gap_reversal_strength',
        'reversal_consistency',
        'range_expansion',
        'directional_expansion',
        'expansion_persistence',
        'asymmetry_momentum_alignment',
        'volume_confirmed_momentum'
    ]
    
    # Apply cross-sectional ranking and combine
    ranked_components = {}
    for component in components:
        if component in data.columns:
            ranked_components[component] = data.groupby(data.index).apply(
                lambda x: cross_sectional_rank(x[component])
            )
    
    # Final factor combination with weights
    factor_weights = {
        'asymmetry_ratio': 0.15,
        'volume_polarity': 0.12,
        'gap_reversal_strength': 0.13,
        'reversal_consistency': 0.10,
        'range_expansion': 0.12,
        'directional_expansion': 0.10,
        'expansion_persistence': 0.08,
        'asymmetry_momentum_alignment': 0.10,
        'volume_confirmed_momentum': 0.10
    }
    
    # Calculate weighted factor
    for date in data.index:
        day_factors = []
        for component, weight in factor_weights.items():
            if component in ranked_components and date in ranked_components[component].index:
                day_factors.append(ranked_components[component].loc[date] * weight)
        
        if day_factors:
            factor.loc[date] = sum(day_factors)
    
    # Fill any remaining NaN values with neutral score (0.5)
    factor = factor.fillna(0.5)
    
    return factor
