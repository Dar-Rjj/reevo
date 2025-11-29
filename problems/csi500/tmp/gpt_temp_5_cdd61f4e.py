import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Range Efficiency Divergence Factor
    # Calculate Intraday Range Efficiency
    df['intraday_range'] = df['high'] - df['low']
    df['range_volume_ratio'] = df['intraday_range'] / (df['volume'] + 1e-8)
    
    # Track efficiency momentum
    df['efficiency_ma5'] = df['range_volume_ratio'].rolling(window=5).mean()
    df['efficiency_ma10'] = df['range_volume_ratio'].rolling(window=10).mean()
    df['efficiency_momentum'] = df['efficiency_ma5'] - df['efficiency_ma10']
    df['efficiency_acceleration'] = df['efficiency_momentum'].diff()
    
    # Analyze Range Expansion Quality
    df['range_expansion'] = df['intraday_range'].pct_change()
    df['volume_expansion'] = df['volume'].pct_change()
    df['expansion_alignment'] = df['range_expansion'] * df['volume_expansion']
    
    # Compute expansion sustainability
    df['expansion_persistence'] = df['expansion_alignment'].rolling(window=3).apply(
        lambda x: np.sum(x > 0) / len(x) if len(x) == 3 else np.nan
    )
    
    # Compute Efficiency-Momentum Divergence
    df['price_momentum'] = df['close'].pct_change(5)
    df['efficiency_momentum_5d'] = df['range_volume_ratio'].pct_change(5)
    df['divergence_strength'] = df['efficiency_momentum_5d'] - df['price_momentum']
    
    # Track divergence persistence
    df['divergence_persistence'] = df['divergence_strength'].rolling(window=5).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 5 else np.nan
    )
    
    # Generate Divergence Signal
    df['divergence_signal'] = (df['divergence_strength'] * df['divergence_persistence'] * 
                              df['expansion_persistence'] * df['efficiency_acceleration'])
    
    # Amount Concentration Momentum Factor
    # Calculate Transaction Concentration Dynamics
    df['amount_ma5'] = df['amount'].rolling(window=5).mean()
    df['amount_ma10'] = df['amount'].rolling(window=10).mean()
    df['concentration_ratio'] = df['amount_ma5'] / (df['amount_ma10'] + 1e-8)
    df['concentration_momentum'] = df['concentration_ratio'].pct_change(3)
    
    # Measure Concentration-Price Interaction
    df['price_reaction'] = df['close'].pct_change()
    df['concentration_impact'] = df['concentration_ratio'] * df['price_reaction']
    df['impact_persistence'] = df['concentration_impact'].rolling(window=3).apply(
        lambda x: np.corrcoef(x, range(len(x)))[0,1] if len(x) == 3 and not np.isnan(x).any() else np.nan
    )
    
    # Incorporate Volume Confirmation
    df['volume_amount_corr'] = df['volume'].rolling(window=5).corr(df['amount'])
    df['confirmation_strength'] = df['volume_amount_corr'].rolling(window=3).mean()
    
    # Generate Concentration Momentum Signal
    df['concentration_signal'] = (df['concentration_momentum'] * df['impact_persistence'] * 
                                 df['confirmation_strength'])
    
    # Price Level Absorption Efficiency Factor
    # Calculate Gap Absorption Dynamics
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_absorption'] = (df['close'] - df['open']) / (df['open'] - df['close'].shift(1) + 1e-8)
    df['absorption_efficiency'] = df['gap_absorption'] * (1 - abs(df['overnight_gap']))
    
    # Incorporate Price Level Memory
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['proximity_to_high'] = (df['high'] - df['prev_high']) / df['prev_high']
    df['proximity_to_low'] = (df['low'] - df['prev_low']) / df['prev_low']
    df['level_attraction'] = (abs(df['proximity_to_high']) + abs(df['proximity_to_low'])) / 2
    
    # Compute Absorption-Level Alignment
    df['absorption_alignment'] = df['absorption_efficiency'] * (1 - df['level_attraction'])
    
    # Generate Absorption Efficiency Signal
    df['absorption_signal'] = df['absorption_alignment'] * df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-8)
    
    # Session Transition Concentration Factor
    # Simplified session analysis using intraday patterns
    df['morning_range'] = (df['high'] - df['low']).rolling(window=3).mean()
    df['afternoon_range'] = (df['high'] - df['low']).shift(-1).rolling(window=3).mean()
    df['session_transition'] = df['afternoon_range'] / (df['morning_range'] + 1e-8)
    
    # Compute Transition Efficiency
    df['morning_momentum'] = (df['close'] - df['open']) / df['open']
    df['afternoon_momentum'] = (df['close'].shift(-1) - df['open'].shift(-1)) / df['open'].shift(-1)
    df['momentum_transfer'] = df['morning_momentum'] * df['afternoon_momentum']
    
    # Incorporate Volume Flow Analysis
    df['morning_volume'] = df['volume'].rolling(window=3).mean()
    df['afternoon_volume'] = df['volume'].shift(-1).rolling(window=3).mean()
    df['volume_transition'] = df['afternoon_volume'] / (df['morning_volume'] + 1e-8)
    
    # Generate Transition Signal
    df['transition_signal'] = (df['session_transition'] * df['momentum_transfer'] * 
                              df['volume_transition'] * df['concentration_ratio'])
    
    # Volatility Compression Breakout Factor
    # Calculate Volatility Compression Dynamics
    df['volatility_5d'] = df['intraday_range'].rolling(window=5).std()
    df['volatility_10d'] = df['intraday_range'].rolling(window=10).std()
    df['compression_ratio'] = df['volatility_5d'] / (df['volatility_10d'] + 1e-8)
    
    # Compute Breakout Probability
    df['volume_accumulation'] = df['volume'] / df['volume'].rolling(window=10).mean()
    df['breakout_probability'] = (1 - df['compression_ratio']) * df['volume_accumulation']
    
    # Combine Compression and Regime Signals
    df['volatility_regime'] = df['volatility_10d'].pct_change(5)
    df['breakout_score'] = df['breakout_probability'] * (1 + df['volatility_regime'])
    
    # Generate Breakout Prediction Signal
    df['breakout_signal'] = (df['breakout_score'] * df['concentration_ratio'] * 
                            df['absorption_efficiency'])
    
    # Combine all factors with equal weighting
    factors = ['divergence_signal', 'concentration_signal', 'absorption_signal', 
               'transition_signal', 'breakout_signal']
    
    # Normalize each factor
    for factor in factors:
        df[f'{factor}_norm'] = (df[factor] - df[factor].rolling(window=20).mean()) / (df[factor].rolling(window=20).std() + 1e-8)
    
    # Final combined factor
    df['combined_factor'] = sum(df[f'{factor}_norm'] for factor in factors) / len(factors)
    
    return df['combined_factor']
