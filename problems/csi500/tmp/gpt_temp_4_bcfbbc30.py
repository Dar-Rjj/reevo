import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Exhaustion with Liquidity Absorption factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price metrics
    df['intraday_high_gain'] = (df['high'] - df['open']) / df['open']
    df['intraday_low_loss'] = (df['low'] - df['open']) / df['open']
    df['close_retracement'] = (df['close'] - df['open']) / df['open']
    
    # Rolling percentiles for momentum extremes (10-day window)
    df['high_gain_percentile'] = df['intraday_high_gain'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) if len(x) >= 5 else np.nan, raw=False
    )
    df['low_loss_percentile'] = df['intraday_low_loss'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] < np.percentile(x, 20)) if len(x) >= 5 else np.nan, raw=False
    )
    
    # Momentum exhaustion signals
    df['exhaustion_signal'] = (
        (df['high_gain_percentile'] == 1) & (df['close_retracement'] < df['intraday_high_gain'] * 0.5) |
        (df['low_loss_percentile'] == 1) & (df['close_retracement'] > df['intraday_low_loss'] * 0.5)
    ).astype(float)
    
    # Liquidity absorption analysis
    # Volume concentration (simplified as volume/price range ratio)
    df['price_range'] = (df['high'] - df['low']) / df['open']
    df['volume_concentration'] = df['volume'] / (df['price_range'] + 1e-8)
    
    # Rolling volume concentration percentiles
    df['volume_concentration_percentile'] = df['volume_concentration'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) if len(x) >= 5 else np.nan, raw=False
    )
    
    # Liquidity exhaustion score
    df['liquidity_exhaustion'] = (
        (df['volume_concentration_percentile'] == 1) & 
        (df['volume'] < df['volume'].rolling(window=10, min_periods=5).mean())
    ).astype(float)
    
    # Market regime context
    df['volatility'] = df['close'].pct_change().rolling(window=10, min_periods=5).std()
    df['volatility_regime'] = df['volatility'] > df['volatility'].rolling(window=20, min_periods=10).median()
    
    # Combine signals with regime adjustment
    factor = (
        df['exhaustion_signal'] * 
        df['liquidity_exhaustion'] * 
        (1 + 0.5 * df['volatility_regime'].astype(float))
    )
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
