import pandas as pd
def heuristics_v2(df):
    # Raw Intraday Momentum Signal: (High - Low) / Close
    raw_momentum = (df['high'] - df['low']) / df['close']
    
    # Volume Confirmation Component
    # Calculate 20-day average volume (using only past data)
    avg_volume_20d = df['volume'].rolling(window=20, min_periods=1).mean()
    # Daily volume ratio (clipped between 0.5 and 2.0)
    volume_ratio = df['volume'] / avg_volume_20d
    volume_confirmation = volume_ratio.clip(lower=0.5, upper=2.0)
    
    # Combined Signal Generation
    combined_signal = raw_momentum * volume_confirmation
    # 5-day EMA of combined signal (using only past data)
    ema_5d = combined_signal.ewm(span=5, adjust=False).mean()
    
    # Final Factor Construction
    # Rank the EMA signal (using only past data for ranking)
    ranked_signal = ema_5d.rolling(window=len(ema_5d), min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    # Standardize the ranked signal (z-score)
    factor = (ranked_signal - ranked_signal.mean()) / ranked_signal.std()
    
    return factor
