import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Fractal Efficiency Alpha Factor
    Combines fractal volatility patterns, price efficiency, volume asymmetry, 
    session boundary effects, and range expansion dynamics
    """
    data = df.copy()
    
    # Multi-Scale Volatility Fractal Analysis
    # Short-term volatility (3-day High-Low range)
    data['short_term_vol'] = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    
    # Medium-term volatility (10-day High-Low range)
    data['medium_term_vol'] = data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()
    
    # Multi-Scale Volatility Ratio
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    data['vol_ratio'] = data['vol_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Range Persistence (5-day range percentile rank)
    data['daily_range'] = data['high'] - data['low']
    data['range_persistence'] = data['daily_range'].rolling(window=5).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Fractal Transition Momentum
    data['fractal_transition'] = data['vol_ratio'] * data['range_persistence']
    
    # Fractal Price Movement Efficiency
    data['price_efficiency'] = (data['close'] - data['open']).abs() / (data['high'] - data['low'])
    data['price_efficiency'] = data['price_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Multi-Regime Efficiency Ratio (5-day rolling)
    data['optimal_path'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()).abs()
    data['actual_path'] = data['close'].diff(5).abs()
    data['efficiency_ratio'] = data['optimal_path'] / data['actual_path']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Fractal Efficiency-Volatility Interaction
    data['fractal_efficiency'] = data['efficiency_ratio'] * data['fractal_transition']
    
    # Volume-Price Asymmetry Analysis
    data['price_direction'] = np.where(data['close'] > data['open'], 1, -1)
    data['volume_return'] = data['price_direction'] * (data['close'] - data['open']) / data['close'].shift(1)
    data['volume_asymmetry'] = data['volume'] * data['volume_return']
    data['volume_asymmetry_ma'] = data['volume_asymmetry'].rolling(window=5).mean()
    
    # Volume-Confirmed Fractal Transition
    data['volume_confirmed_fractal'] = data['fractal_transition'] * data['volume_asymmetry_ma']
    
    # Session Boundary Fractal Analysis
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['closing_momentum'] = (data['close'] - data['open']) / data['open']
    
    # Session boundary volatility clustering
    data['session_volatility'] = data['daily_range'].rolling(window=3).std()
    data['boundary_effect'] = data['opening_gap'] * data['closing_momentum'] * data['session_volatility']
    
    # Range Expansion-Compression Analysis
    data['range_1d'] = data['daily_range']
    data['range_3d'] = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    data['range_5d'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    
    # Fractal range convergence
    data['range_convergence'] = (data['range_1d'] / data['range_3d'] + 
                                data['range_3d'] / data['range_5d']) / 2
    data['range_convergence'] = data['range_convergence'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Volume concentration fractal patterns
    data['volume_ma'] = data['volume'].rolling(window=5).mean()
    data['volume_concentration'] = data['volume'] / data['volume_ma']
    
    # Composite Fractal Efficiency Alpha Construction
    # Core component: Fractal transition with efficiency and volume confirmation
    core_component = (data['fractal_efficiency'] * 
                     data['volume_confirmed_fractal'] * 
                     data['range_convergence'])
    
    # Session boundary enhancement
    session_enhancement = data['boundary_effect'] * data['volume_concentration']
    
    # Range expansion confirmation
    range_confirmation = data['range_convergence'] * data['fractal_transition']
    
    # Final composite factor
    fractal_alpha = (core_component * 
                    (1 + session_enhancement) * 
                    (1 + range_confirmation))
    
    # Directional adjustment based on price momentum
    price_momentum = data['close'].pct_change(3)
    directional_adjustment = np.sign(price_momentum) * np.abs(fractal_alpha)
    
    # Final factor with momentum alignment
    final_factor = directional_adjustment * np.abs(fractal_alpha)
    
    return final_factor
