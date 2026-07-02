import pandas as pd
def heuristics_v2(data):
    # Price Impact Ratio calculation
    delta_close = data['close'].diff()
    abs_delta_close = delta_close.abs()
    rolling_volume_sum = data['volume'].rolling(window=5, min_periods=1).sum()
    price_impact_ratio = abs_delta_close / rolling_volume_sum
    # Cross-sectional normalization
    normalized_pir = price_impact_ratio.groupby(price_impact_ratio.index).transform(lambda x: (x - x.mean()) / x.std())
    
    # Order Flow Divergence calculation
    ema_high = data['high'].ewm(span=3, adjust=False).mean()
    ema_low = data['low'].ewm(span=3, adjust=False).mean()
    order_flow_divergence = ema_high - ema_low
    # Rolling rank
    ranked_ofd = order_flow_divergence.rolling(window=10, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine factors
    factor = normalized_pir + ranked_ofd
    return factor
