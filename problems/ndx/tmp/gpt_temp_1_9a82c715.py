import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 1. Multi-resolution momentum with dynamic regime weights
    mom_short = df['close'].pct_change(3) * (1 + df['volume'].pct_change(3))
    mom_mid = df['close'].pct_change(7) * np.sqrt(1 + df['volume'].pct_change(7))
    mom_long = df['close'].pct_change(21) * (1 + np.log1p(df['volume'].pct_change(21)))
    
    # Volatility-adaptive momentum blending
    vol_ratio = df['high'].rolling(5).std() / df['low'].rolling(5).std()
    momentum_core = (
        (0.4 * np.tanh(vol_ratio) * mom_short) 
        + (0.5 * mom_mid / (1 + np.abs(vol_ratio - 1))) 
        + (0.1 * mom_long * (2 - vol_ratio))
    )
    
    # 2. Intraday pressure signals with volume confirmation
    price_range = (df['high'] - df['low']) / df['close']
    upside_pressure = (df['close'] - df['open']) / (price_range + 1e-7) * np.log1p(df['volume'])
    downside_rejection = (df['open'] - df['low']) / (price_range + 1e-7) * np.sqrt(df['volume'])
    intraday_signal = upside_pressure - downside_rejection
    
    # 3. Asymmetric liquidity adjustment
    liquidity = np.log1p(df['volume'].rolling(10).mean())
    adv_ratio = df['volume'] / df['volume'].rolling(21).mean()
    liquidity_boost = np.where(adv_ratio > 1, 
                             np.sqrt(adv_ratio), 
                             np.power(adv_ratio, 1.5))
    
    # 4. Non-linear combination with regime awareness
    combined = (
        0.7 * momentum_core * np.tanh(liquidity_boost)
        + 0.3 * intraday_signal * np.sqrt(liquidity)
    )
    
    # 5. Dynamic smoothing with IQR-based outlier control
    rolling_iqr = combined.rolling(21).quantile(0.75) - combined.rolling(21).quantile(0.25)
    smoothed = combined.rolling(5).apply(lambda x: np.median(x[x.between(x.quantile(0.25), x.quantile(0.75))]))
    
    # Final volatility-scaled output
    return smoothed / (combined.rolling(10).std() + 1e-7)
