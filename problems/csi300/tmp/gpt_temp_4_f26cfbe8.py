def heuristics_v2(df):
    # Calculate intraday price range
    price_range = df['high'] - df['low']
    
    # Normalize price range by opening price
    normalized_range = price_range / df['open']
    
    # Calculate volume concentration (first hour vs last hour)
    # Assuming first hour is represented by the first 1/6 of daily data points (4 hours in 24-hour market)
    # and last hour is the last 1/6 of daily data points
    daily_points = len(df) // (df.index.normalize().nunique())
    first_hour_points = daily_points // 6
    last_hour_points = daily_points // 6
    
    # Group by date to calculate daily metrics
    grouped = df.groupby(df.index.normalize())
    
    # Calculate first hour and last hour volume for each day
    first_hour_vol = grouped['volume'].apply(lambda x: x.iloc[:first_hour_points].sum())
    last_hour_vol = grouped['volume'].apply(lambda x: x.iloc[-last_hour_points:].sum())
    
    # Calculate volume concentration ratio
    vol_concentration = (first_hour_vol + last_hour_vol) / grouped['volume'].sum()
    
    # Reindex to match original dataframe
    vol_concentration = vol_concentration.reindex(df.index.normalize()).ffill()
    
    # Combine components
    factor = normalized_range * vol_concentration
    
    # Apply 3-day moving average (using only past and current data)
    factor_smoothed = factor.rolling(window=3, min_periods=1).mean()
    
    return factor_smoothed
