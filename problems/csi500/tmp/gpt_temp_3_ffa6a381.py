import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Scale Momentum Efficiency with Microstructural Confirmation
    Combines multi-timeframe momentum efficiency with volume absorption and price elasticity validation
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Efficiency
    # Intraday efficiency: (Close - Open) / (High - Low)
    intraday_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Overnight efficiency: (Open - Previous Close) / (High - Low)
    prev_close = data['close'].shift(1)
    overnight_efficiency = (data['open'] - prev_close) / (data['high'] - data['low'])
    overnight_efficiency = overnight_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Cross-scale efficiency divergence analysis
    efficiency_divergence = intraday_efficiency - overnight_efficiency.rolling(window=5).mean()
    
    # Primary efficiency score (weighted combination)
    primary_efficiency = 0.6 * intraday_efficiency + 0.4 * overnight_efficiency + 0.2 * efficiency_divergence
    
    # Volume Absorption Confirmation
    # Key level absorption: Volume at (High + Low)/2 / Average volume
    key_level = (data['high'] + data['low']) / 2
    # Use rolling average volume to avoid lookahead bias
    avg_volume_5d = data['volume'].rolling(window=5).mean()
    volume_absorption = data['volume'] / avg_volume_5d
    
    # Absorption-momentum interaction patterns
    absorption_momentum_interaction = volume_absorption * np.sign(primary_efficiency)
    
    # Volume absorption multiplier (normalized)
    volume_multiplier = volume_absorption.rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    volume_multiplier = volume_multiplier.clip(-2, 2)  # Cap extreme values
    
    # Price Elasticity Validation
    # Momentum response elasticity: |Efficiency change| / Volume shock
    efficiency_change = primary_efficiency.diff().abs()
    volume_shock = (data['volume'] / data['volume'].rolling(window=10).mean() - 1).abs()
    
    # Avoid division by zero
    volume_shock_adj = volume_shock.replace(0, np.nan)
    price_elasticity = efficiency_change / volume_shock_adj
    price_elasticity = price_elasticity.replace([np.inf, -np.inf], np.nan)
    
    # Elasticity-momentum confirmation states
    elasticity_confirmation = np.where(
        (primary_efficiency > 0) & (price_elasticity > price_elasticity.rolling(window=5).mean()), 1,
        np.where(
            (primary_efficiency < 0) & (price_elasticity > price_elasticity.rolling(window=5).mean()), -1, 0
        )
    )
    
    # Elasticity confirmation filter (normalized)
    elasticity_filter = pd.Series(elasticity_confirmation, index=data.index).rolling(window=3).mean()
    
    # Factor Construction
    # Final factor: Efficiency × Absorption × Elasticity
    factor = primary_efficiency * volume_multiplier * elasticity_filter
    
    # Clean and normalize the final factor
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor
