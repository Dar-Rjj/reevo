import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency Components
    df['true_range'] = df['high'] - df['low']
    df['effective_movement'] = np.abs(df['close'] - df['open'])
    df['efficiency_ratio'] = df['effective_movement'] / df['true_range'].replace(0, np.nan)
    
    # Gap Sustainability Analysis
    df['gap'] = df['open'] - df['close'].shift(1)
    df['first_hour_range'] = (df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()).shift(-2)
    df['gap_sustainability'] = np.abs(df['first_hour_range']) / (np.abs(df['gap']) + 1e-8)
    df['price_rejection'] = np.where(
        df['gap'] > 0,
        (df['high'] - df['open']) / df['gap'],
        (df['open'] - df['low']) / np.abs(df['gap'])
    )
    
    # Volume-Efficiency Confirmation
    df['volume_rank'] = df['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['efficiency_rank'] = df['efficiency_ratio'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['volume_efficiency_alignment'] = np.where(
        df['efficiency_rank'] > 0.7,
        df['volume_rank'] * df['efficiency_rank'],
        -df['volume_rank'] * (1 - df['efficiency_rank'])
    )
    
    # Opening Volume Dynamics
    df['opening_volume_surge'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['gap_volume_pattern'] = np.where(
        df['gap'] > 0,
        df['volume'] * df['efficiency_ratio'],
        -df['volume'] * df['efficiency_ratio']
    )
    
    # Volume-Pressure Equilibrium
    df['volume_pressure'] = df['volume'] * df['effective_movement'] / df['true_range'].replace(0, np.nan)
    df['pressure_imbalance'] = (df['volume_pressure'] - df['volume_pressure'].rolling(window=10).mean()) / df['volume_pressure'].rolling(window=10).std()
    
    # Amount-Price Coherence
    df['price_per_amount'] = df['effective_movement'] / (df['amount'] + 1e-8)
    df['amount_concentration'] = df['amount'] / df['amount'].rolling(window=20).mean()
    
    # Large Amount Reaction Analysis
    df['amount_efficiency'] = df['effective_movement'] / (df['amount'] + 1e-8)
    df['accumulation_signal'] = np.where(
        (df['amount_concentration'] > 1.5) & (df['efficiency_ratio'] < 0.3),
        df['amount_concentration'] * (1 - df['efficiency_ratio']),
        0
    )
    df['manipulation_signal'] = np.where(
        (df['amount_concentration'] < 0.7) & (df['efficiency_ratio'] > 0.7),
        -df['efficiency_ratio'] * (1 - df['amount_concentration']),
        0
    )
    
    # Amount-Price Divergence
    df['amount_price_divergence'] = (df['amount_efficiency'] - df['amount_efficiency'].rolling(window=10).mean()) / df['amount_efficiency'].rolling(window=10).std()
    
    # Close Position Dynamics
    df['close_position'] = (df['close'] - df['low']) / (df['true_range'] + 1e-8)
    df['closing_momentum'] = df['close_position'].diff(3)
    df['multi_day_strength'] = df['close_position'].rolling(window=5).mean()
    
    # Range-Momentum Divergence
    df['range_momentum'] = df['true_range'].pct_change(3)
    df['price_momentum'] = df['close'].pct_change(3)
    df['range_price_divergence'] = df['range_momentum'] - df['price_momentum']
    
    # Volatility Regime Context
    df['range_expansion'] = df['true_range'] / df['true_range'].rolling(window=20).mean()
    df['volume_regime'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Transition Quality Assessment
    df['clean_transition'] = np.where(
        (df['range_expansion'] > 1.2) & (df['volume_regime'] > 1.1) & (df['efficiency_ratio'] > 0.6),
        df['range_expansion'] * df['volume_regime'] * df['efficiency_ratio'],
        0
    )
    df['failed_transition'] = np.where(
        (df['range_expansion'] > 1.2) & (df['volume_regime'] < 0.9) & (df['efficiency_ratio'] < 0.4),
        -df['range_expansion'] * (1 - df['volume_regime']) * (1 - df['efficiency_ratio']),
        0
    )
    
    # Composite Factor Calculation
    factors = [
        df['efficiency_ratio'].fillna(0),
        df['gap_sustainability'].fillna(0),
        df['price_rejection'].fillna(0),
        df['volume_efficiency_alignment'].fillna(0),
        df['opening_volume_surge'].fillna(0),
        df['gap_volume_pattern'].fillna(0),
        df['pressure_imbalance'].fillna(0),
        df['price_per_amount'].fillna(0),
        df['amount_concentration'].fillna(0),
        df['accumulation_signal'].fillna(0),
        df['manipulation_signal'].fillna(0),
        df['amount_price_divergence'].fillna(0),
        df['close_position'].fillna(0),
        df['closing_momentum'].fillna(0),
        df['multi_day_strength'].fillna(0),
        df['range_price_divergence'].fillna(0),
        df['clean_transition'].fillna(0),
        df['failed_transition'].fillna(0)
    ]
    
    # Normalize and combine factors
    normalized_factors = []
    for factor in factors:
        if len(factor) > 20:
            normalized = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
            normalized_factors.append(normalized.fillna(0))
        else:
            normalized_factors.append(factor * 0)
    
    # Weighted combination (equal weights for demonstration)
    composite_factor = sum(normalized_factors) / len(normalized_factors)
    
    return composite_factor
