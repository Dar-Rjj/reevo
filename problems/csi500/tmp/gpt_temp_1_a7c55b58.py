import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Transition with Price-Volume Asymmetry factor
    """
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # 1. Volatility Regime Classification
    # Historical volatility calculation
    vol_20d = returns.rolling(window=20).std()
    vol_percentile = vol_20d.rolling(window=60).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) if len(x.dropna()) > 0 else np.nan, raw=False)
    
    # Regime classification: 1 for high volatility, 0 for low volatility
    high_vol_regime = (vol_percentile > 0.7).astype(int)
    
    # Regime transition detection
    regime_shift = high_vol_regime.diff()
    low_to_high = (regime_shift == 1).astype(int)
    high_to_low = (regime_shift == -1).astype(int)
    
    # Regime persistence
    regime_duration = high_vol_regime.groupby((high_vol_regime != high_vol_regime.shift()).cumsum()).cumcount() + 1
    
    # 2. Price-Volume Asymmetry Measurement
    # Directional price movement
    up_day = (returns > 0).astype(int)
    down_day = (returns < 0).astype(int)
    price_magnitude = abs(returns)
    
    # Volume response asymmetry
    volume_up = df['volume'] * up_day
    volume_down = df['volume'] * down_day
    
    # Rolling volume asymmetry ratio
    vol_up_avg = volume_up.rolling(window=10).mean()
    vol_down_avg = volume_down.rolling(window=10).mean()
    volume_asymmetry = (vol_up_avg - vol_down_avg) / (vol_up_avg + vol_down_avg + 1e-8)
    
    # Regime-dependent asymmetry
    low_vol_asymmetry = volume_asymmetry * (1 - high_vol_regime)
    high_vol_asymmetry = volume_asymmetry * high_vol_regime
    
    # 3. Intraday Price Efficiency Analysis
    # Opening price efficiency
    daily_range = (df['high'] - df['low']) / df['close']
    opening_gap = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    opening_efficiency = opening_gap / (daily_range + 1e-8)
    
    # Closing price efficiency
    close_to_high = (df['high'] - df['close']) / df['close']
    close_to_low = (df['close'] - df['low']) / df['close']
    closing_efficiency = (close_to_high - close_to_low) / (daily_range + 1e-8)
    
    # Regime impact on efficiency
    low_vol_efficiency = opening_efficiency * (1 - high_vol_regime)
    high_vol_efficiency = closing_efficiency * high_vol_regime
    
    # 4. Multi-Timeframe Signal Integration
    # Regime transition signals
    transition_signal = low_to_high - high_to_low
    
    # Asymmetry-based signals
    asymmetry_signal = volume_asymmetry.rolling(window=5).mean()
    
    # Efficiency timing signals
    efficiency_signal = (opening_efficiency.rolling(window=5).mean() - 
                        closing_efficiency.rolling(window=5).mean())
    
    # Composite alpha factor
    # Weight signals by regime persistence
    regime_weight = 1 / (1 + np.exp(-regime_duration / 10))
    
    # Combine signals with regime-adaptive weighting
    composite_alpha = (
        transition_signal * 0.3 +
        asymmetry_signal * 0.4 * regime_weight +
        efficiency_signal * 0.3 * (1 - regime_weight)
    )
    
    # Normalize the final factor
    alpha_factor = (composite_alpha - composite_alpha.rolling(window=20).mean()) / composite_alpha.rolling(window=20).std()
    
    return alpha_factor
