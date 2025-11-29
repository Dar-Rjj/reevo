import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate all components
    for i in range(len(data)):
        if i < 5:  # Need at least 5 days for some calculations
            factor.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        # 1. Intraday Momentum Amplitude Factor
        intraday_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        price_change = current_data['close'].iloc[-1] - current_data['open'].iloc[-1]
        
        if intraday_range > 0:
            normalized_momentum = (price_change / intraday_range) * current_data['volume'].iloc[-1]
        else:
            normalized_momentum = 0
            
        # 2. High-Low Breakout Efficiency
        high_breakout = current_data['high'].iloc[-1] - current_data['open'].iloc[-1]
        low_breakout = current_data['open'].iloc[-1] - current_data['low'].iloc[-1]
        
        close_to_high = current_data['close'].iloc[-1] / current_data['high'].iloc[-1] - 1
        close_to_low = current_data['close'].iloc[-1] / current_data['low'].iloc[-1] - 1
        
        if close_to_high > close_to_low:
            breakout_magnitude = high_breakout
        else:
            breakout_magnitude = low_breakout
            
        breakout_efficiency = breakout_magnitude * current_data['volume'].iloc[-1]
        
        # 3. Volume-Adjusted Price Acceleration
        # Calculate daily returns
        returns = current_data['close'].pct_change().dropna()
        if len(returns) >= 3:
            first_diff = returns.diff().iloc[-1]
            second_diff = returns.diff().diff().iloc[-1]
            price_acceleration = second_diff if not np.isnan(second_diff) else 0
        else:
            price_acceleration = 0
            
        # Volume momentum
        volume_change = current_data['volume'].pct_change().iloc[-1]
        if np.isnan(volume_change):
            volume_change = 0
            
        volume_adjusted_accel = price_acceleration * volume_change * abs(price_change)
        
        # 4. Opening Gap Persistence Factor
        if i > 0:
            opening_gap = abs(current_data['open'].iloc[-1] / current_data['close'].iloc[-2] - 1)
            intraday_move = current_data['close'].iloc[-1] / current_data['open'].iloc[-1] - 1
            gap_sign = np.sign(current_data['open'].iloc[-1] / current_data['close'].iloc[-2] - 1)
            
            if opening_gap > 0:
                persistence_ratio = intraday_move / opening_gap * gap_sign
            else:
                persistence_ratio = 0
        else:
            persistence_ratio = 0
            
        trading_intensity = current_data['volume'].iloc[-1] * current_data['amount'].iloc[-1]
        gap_persistence = persistence_ratio * trading_intensity
        
        # 5. Volatility-Regulated Trend Strength
        # 5-day rolling slope
        recent_prices = current_data['close'].iloc[-5:]
        if len(recent_prices) == 5:
            x = np.arange(5)
            slope = np.polyfit(x, recent_prices.values, 1)[0]
        else:
            slope = 0
            
        # Realized volatility (using high-low range and close-to-close returns)
        recent_high_low = (current_data['high'].iloc[-5:] - current_data['low'].iloc[-5:]).mean()
        recent_returns_vol = returns.iloc[-5:].std() if len(returns) >= 5 else 0
        realized_vol = (recent_high_low + abs(recent_returns_vol)) / 2
        
        # Volume trend
        volume_trend = current_data['volume'].iloc[-5:].pct_change().mean()
        if np.isnan(volume_trend):
            volume_trend = 0
            
        if realized_vol > 0:
            regulated_trend = (slope / realized_vol) * volume_trend
        else:
            regulated_trend = 0
            
        # Combine all factors with equal weights
        combined_factor = (
            normalized_momentum +
            breakout_efficiency +
            volume_adjusted_accel +
            gap_persistence +
            regulated_trend
        )
        
        factor.iloc[i] = combined_factor
    
    return factor
