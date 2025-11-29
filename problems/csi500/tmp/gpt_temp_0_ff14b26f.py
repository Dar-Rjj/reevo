import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Range Efficiency Divergence Factor
    df['range_volume_efficiency'] = (df['high'] - df['low']) / (df['volume'] + 1e-8)
    df['efficiency_momentum'] = df['range_volume_efficiency'].rolling(window=5).mean() - df['range_volume_efficiency'].rolling(window=20).mean()
    df['price_momentum'] = df['close'].pct_change(5) - df['close'].pct_change(20)
    df['efficiency_divergence'] = df['efficiency_momentum'] - df['price_momentum']
    df['divergence_strength'] = df['efficiency_divergence'].rolling(window=10).std()
    df['range_efficiency_factor'] = df['efficiency_divergence'] * df['divergence_strength']
    
    # Amount Concentration Momentum Factor
    df['amount_concentration'] = df['amount'] / (df['volume'] + 1e-8)
    df['concentration_momentum'] = df['amount_concentration'].pct_change(3).rolling(window=5).mean()
    df['large_transaction_impact'] = (df['amount'] * df['close']).rolling(window=5).std() / (df['amount'].rolling(window=5).mean() + 1e-8)
    df['concentration_efficiency'] = df['concentration_momentum'] / (df['large_transaction_impact'] + 1e-8)
    df['volume_confirmation'] = df['volume'].pct_change(3).rolling(window=5).mean()
    df['amount_concentration_factor'] = df['concentration_efficiency'] * df['volume_confirmation']
    
    # Price Level Absorption Efficiency Factor
    df['gap_magnitude'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_recovery'] = (df['close'] - df['open']) / (df['open'] - df['close'].shift(1) + 1e-8)
    df['absorption_efficiency'] = df['gap_recovery'].abs() * np.sign(df['gap_magnitude'])
    df['key_level_proximity'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
    df['level_attraction'] = -df['key_level_proximity'].abs()
    df['absorption_factor'] = df['absorption_efficiency'] * df['level_attraction']
    
    # Session Transition Concentration Factor
    df['morning_range'] = (df['high'] - df['low']).rolling(window=5).apply(lambda x: x.iloc[:3].mean() if len(x) >= 3 else np.nan)
    df['afternoon_range'] = (df['high'] - df['low']).rolling(window=5).apply(lambda x: x.iloc[3:].mean() if len(x) >= 5 else np.nan)
    df['transition_smoothness'] = (df['morning_range'] - df['afternoon_range']) / (df['morning_range'] + df['afternoon_range'] + 1e-8)
    df['momentum_transfer'] = df['close'].pct_change().rolling(window=3).std()
    df['concentration_alignment'] = df['amount_concentration'].pct_change(3) * df['transition_smoothness']
    df['volume_flow'] = df['volume'].pct_change(3).rolling(window=5).mean()
    df['session_transition_factor'] = df['concentration_alignment'] * df['volume_flow']
    
    # Volatility Compression Breakout Factor
    df['range_contraction'] = (df['high'] - df['low']).rolling(window=10).std() / (df['high'] - df['low']).rolling(window=30).std()
    df['compression_duration'] = ((df['high'] - df['low']) < (df['high'] - df['low']).rolling(window=20).mean()).rolling(window=10).sum()
    df['volume_accumulation'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
    df['volatility_regime'] = (df['high'] - df['low']).rolling(window=20).std() / (df['high'] - df['low']).rolling(window=60).std()
    df['breakout_probability'] = df['volume_accumulation'] * (1 - df['volatility_regime'])
    df['compression_breakout_factor'] = (1 - df['range_contraction']) * df['compression_duration'] * df['breakout_probability']
    
    # Combine factors with equal weights
    factors = [
        'range_efficiency_factor',
        'amount_concentration_factor', 
        'absorption_factor',
        'session_transition_factor',
        'compression_breakout_factor'
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        normalized = (df[factor] - df[factor].rolling(window=60).mean()) / (df[factor].rolling(window=60).std() + 1e-8)
        combined_factor += normalized
    
    return combined_factor
