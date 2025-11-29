import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        current_idx = data.index.get_loc(date)
        
        # Skip if we don't have enough history
        if current_idx < 20:
            factor_values.loc[date] = np.nan
            continue
            
        current_data = data.iloc[:current_idx+1]  # Only current and past data
        
        # Get current day data
        current_day = current_data.iloc[-1]
        prev_day = current_data.iloc[-2]
        
        # Price-Based Momentum Factors
        # Gap Fade Momentum
        overnight_gap = (current_day['open'] / prev_day['close']) - 1
        intraday_fade = (current_day['close'] / current_day['open']) - 1
        gap_fade_momentum = overnight_gap * intraday_fade
        
        # Range Expansion Momentum
        daily_range = (current_day['high'] / current_day['low']) - 1
        
        # Calculate 5-day average range (only using past data)
        if current_idx >= 5:
            recent_data = current_data.iloc[-5:]
            avg_range = np.mean([(row['high']/row['low'] - 1) for _, row in recent_data.iterrows()])
            range_expansion = daily_range / avg_range if avg_range != 0 else 0
        else:
            range_expansion = 0
            
        close_to_close_return = (current_day['close'] / prev_day['close']) - 1
        range_expansion_momentum = range_expansion * close_to_close_return
        
        # Volume-Price Interaction Factors
        # Volume-Weighted Price Efficiency
        if current_day['high'] != current_day['low']:
            price_efficiency = abs(current_day['close'] - current_day['open']) / (current_day['high'] - current_day['low'])
        else:
            price_efficiency = 0
            
        # Volume change ratio
        if prev_day['volume'] != 0:
            volume_change_ratio = current_day['volume'] / prev_day['volume']
        else:
            volume_change_ratio = 1
            
        # Liquidity adjustment
        liquidity_adjustment = np.log1p(current_day['amount']) if current_day['amount'] > 0 else 0
        volume_weighted_efficiency = price_efficiency * volume_change_ratio * liquidity_adjustment
        
        # Breakout Volume Confirmation
        # 20-day high breaks
        if current_idx >= 20:
            past_20_days = current_data.iloc[-21:-1]  # Previous 20 days excluding current
            twenty_day_high = past_20_days['close'].max()
            high_break = 1 if current_day['close'] > twenty_day_high else 0
        else:
            high_break = 0
            
        # Volume surge (10-day average volume)
        if current_idx >= 10:
            past_10_days = current_data.iloc[-11:-1]  # Previous 10 days excluding current
            avg_volume_10d = past_10_days['volume'].mean()
            volume_surge = current_day['volume'] / avg_volume_10d if avg_volume_10d != 0 else 1
        else:
            volume_surge = 1
            
        break_distance = (current_day['close'] / twenty_day_high - 1) if high_break and current_idx >= 20 else 0
        breakout_volume_confirmation = break_distance * volume_surge * high_break
        
        # Volatility Regime Factors
        # Volatility Acceleration Signal
        daily_volatility = (current_day['high'] - current_day['low']) / prev_day['close']
        
        # 3-day volatility change ratio
        if current_idx >= 3:
            vol_today = daily_volatility
            vol_2d_ago = (current_data.iloc[-3]['high'] - current_data.iloc[-3]['low']) / current_data.iloc[-4]['close']
            vol_3d_ago = (current_data.iloc[-4]['high'] - current_data.iloc[-4]['low']) / current_data.iloc[-5]['close']
            vol_change_ratio = vol_today / ((vol_2d_ago + vol_3d_ago) / 2) if (vol_2d_ago + vol_3d_ago) != 0 else 1
        else:
            vol_change_ratio = 1
            
        price_momentum_direction = np.sign(close_to_close_return)
        volatility_acceleration = vol_change_ratio * price_momentum_direction
        
        # Range Persistence Factor
        # 3-day range autocorrelation
        if current_idx >= 3:
            ranges = []
            for i in range(3):
                day_data = current_data.iloc[-(i+1)]
                day_range = (day_data['high'] - day_data['low']) / day_data['open']
                ranges.append(day_range)
            
            # Simple autocorrelation approximation
            if len(ranges) >= 3:
                range_persistence = (ranges[0] * ranges[1] + ranges[1] * ranges[2]) / 2
            else:
                range_persistence = 0
        else:
            range_persistence = 0
            
        # Volume trend
        if current_idx >= 2:
            volume_trend = (current_day['volume'] / current_data.iloc[-3]['volume']) - 1
        else:
            volume_trend = 0
            
        intraday_return_pattern = intraday_fade
        range_persistence_factor = range_persistence * volume_trend * intraday_return_pattern
        
        # Market Microstructure Factors
        # Opening Auction Efficiency
        prev_range = prev_day['high'] - prev_day['low']
        if prev_range != 0:
            open_vs_prev_range = abs(current_day['open'] - prev_day['close']) / prev_range
        else:
            open_vs_prev_range = 0
            
        # Gap fill speed (using current day's range relative to open)
        if (current_day['high'] - current_day['low']) != 0:
            gap_fill_speed = abs(current_day['close'] - current_day['open']) / (current_day['high'] - current_day['low'])
        else:
            gap_fill_speed = 0
            
        # Opening volume concentration (first hour approximation using total volume)
        opening_volume_concentration = current_day['volume'] / (current_day['amount'] / current_day['close']) if current_day['amount'] > 0 else 0
        opening_auction_efficiency = open_vs_prev_range * gap_fill_speed * opening_volume_concentration
        
        # End-of-Day Pressure
        # Midday high-low approximation (using open and high/low)
        midday_high_low = (current_day['open'] + current_day['high'] + current_day['low']) / 3
        last_hour_move = current_day['close'] / midday_high_low - 1 if midday_high_low != 0 else 0
        
        # Daily volume distribution (using current day's characteristics)
        volume_distribution = current_day['volume'] / (current_day['high'] - current_day['low']) if (current_day['high'] - current_day['low']) != 0 else 0
        
        amount_turnover = current_day['amount'] / current_day['volume'] if current_day['volume'] != 0 else 0
        eod_pressure = last_hour_move * volume_distribution * amount_turnover
        
        # Combine all factors with equal weighting
        combined_factor = (
            gap_fade_momentum +
            range_expansion_momentum +
            volume_weighted_efficiency +
            breakout_volume_confirmation +
            volatility_acceleration +
            range_persistence_factor +
            opening_auction_efficiency +
            eod_pressure
        )
        
        factor_values.loc[date] = combined_factor
    
    return factor_values
