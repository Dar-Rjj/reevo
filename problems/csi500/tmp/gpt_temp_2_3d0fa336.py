import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Component
    # Calculate Momentum Strength
    directional_momentum = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    volatility_adjusted_momentum = (data['high'] - data['low']) / (data['open'] + 1e-8)
    
    # Generate Momentum Signal
    momentum_signal = directional_momentum * volatility_adjusted_momentum
    momentum_signal = np.sign(momentum_signal) * np.abs(momentum_signal)
    
    # Liquidity Divergence Component
    # Calculate Liquidity Dynamics
    amount_change_ratio = data['amount'].pct_change()
    amount_acceleration = amount_change_ratio.diff()
    
    # Create Liquidity-Momentum Interaction
    liquidity_momentum_interaction = momentum_signal * amount_acceleration
    divergence_intensity = liquidity_momentum_interaction.rolling(window=5, min_periods=3).std()
    
    # Price Efficiency Filter
    # Assess Market Efficiency Conditions
    price_autocorr = data['close'].pct_change().rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr() if len(x) >= 3 else np.nan, raw=False
    )
    
    volume_price_corr = data['close'].pct_change().rolling(window=5, min_periods=3).corr(
        data['volume'].pct_change()
    )
    
    # Conditional Signal Refinement
    efficiency_condition = (np.abs(price_autocorr) < 0.1) & (np.abs(volume_price_corr) < 0.2)
    efficiency_adjustment = np.where(efficiency_condition, 0.7, 1.0)
    
    filtered_signal = divergence_intensity * efficiency_adjustment
    
    # Multi-scale Convergence
    # Convergence Detection Across Timeframes
    short_term_convergence = filtered_signal.rolling(window=2, min_periods=1).median()
    medium_term_convergence = filtered_signal.rolling(window=8, min_periods=4).median()
    
    # Final Factor Construction
    convergence_ratio = short_term_convergence / (medium_term_convergence + 1e-8)
    final_factor = convergence_ratio * efficiency_adjustment
    final_factor = np.sign(final_factor) * np.log(np.abs(final_factor) + 1)
    
    return final_factor
