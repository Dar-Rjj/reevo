def heuristics_v2(df):
    # Liquidity Change Signal
    delta_volume = df['volume'].diff(5)  # Delta of volume over a 5-day window
    rolling_rank_delta = delta_volume.rolling(window=20).apply(lambda x: x.rank(pct=True).iloc[-1])  # Rolling rank over 20 days
    zscore_rank = rolling_rank_delta.sub(rolling_rank_delta.mean()).div(rolling_rank_delta.std())  # Z-score of the rolling rank
    
    # Microstructure Confirmation
    buy_volume = df['volume'] * (df['close'] >= df['vwap'])  # Buy volume
    sell_volume = df['volume'] * (df['close'] < df['vwap'])  # Sell volume
    order_flow_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)  # Order flow imbalance
    rolling_order_flow_imbalance = order_flow_imbalance.rolling(window=5).mean()  # Rolling mean over 5 days
    
    volatility_adjusted_return = df['close'].pct_change().div(df['close'].pct_change().rolling(window=10).std())  # Volatility adjusted return
    
    # Combine signals
    factor = zscore_rank + rolling_order_flow_imbalance + volatility_adjusted_return
    return factor
