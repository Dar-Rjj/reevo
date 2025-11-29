import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor mining with four categories:
    1. Intraday Price Efficiency
    2. Volume-Price Dynamics  
    3. Multi-Timeframe Volatility
    4. Overnight Reversal Patterns
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        current_data = df.loc[:date].copy()
        
        if len(current_data) < 20:  # Minimum data requirement
            result.loc[date] = 0
            continue
            
        # 1. Intraday Price Efficiency
        # Opening Gap vs Range Ratio
        if len(current_data) >= 2:
            prev_close = current_data['close'].iloc[-2]
            open_price = current_data['open'].iloc[-1]
            high_price = current_data['high'].iloc[-1]
            low_price = current_data['low'].iloc[-1]
            
            opening_gap = (open_price - prev_close) / prev_close
            daily_range = (high_price - low_price) / prev_close if prev_close > 0 else 0
            
            gap_range_ratio = opening_gap / daily_range if daily_range != 0 else 0
            
            # Price Path Optimality Coefficient
            close_price = current_data['close'].iloc[-1]
            optimal_move = abs(close_price - open_price)
            actual_distance = 0
            prev_price = open_price
            
            # Simplified path calculation using OHLC
            for price in [high_price, low_price, close_price]:
                actual_distance += abs(price - prev_price)
                prev_price = price
                
            path_efficiency = optimal_move / actual_distance if actual_distance > 0 else 1
        else:
            gap_range_ratio = 0
            path_efficiency = 1
        
        # 2. Volume-Price Dynamics
        # Amount Impact Deviation
        if len(current_data) >= 10:
            recent_volume = current_data['volume'].tail(10)
            recent_amount = current_data['amount'].tail(10)
            recent_returns = current_data['close'].pct_change().tail(10).fillna(0)
            
            avg_trade_size = (recent_amount / recent_volume).replace([np.inf, -np.inf], 0).fillna(0)
            amount_impact = avg_trade_size * abs(recent_returns)
            
            amount_deviation = amount_impact.std() / (amount_impact.mean() + 1e-8)
            
            # Volume Elasticity Anomaly
            volume_changes = recent_volume.pct_change().fillna(0)
            price_changes = abs(recent_returns)
            
            # Calculate volume elasticity (sensitivity of volume to price moves)
            valid_data = (price_changes > price_changes.quantile(0.3)) & (volume_changes != 0)
            if valid_data.sum() > 3:
                elasticity = (volume_changes[valid_data] / price_changes[valid_data]).median()
            else:
                elasticity = 0
        else:
            amount_deviation = 0
            elasticity = 0
        
        # 3. Multi-Timeframe Volatility
        # Range Convergence Ratio
        if len(current_data) >= 20:
            short_range = (current_data['high'].tail(5) - current_data['low'].tail(5)) / current_data['close'].tail(5).shift(1).fillna(method='ffill')
            long_range = (current_data['high'].tail(20) - current_data['low'].tail(20)) / current_data['close'].tail(20).shift(1).fillna(method='ffill')
            
            range_convergence = short_range.mean() / (long_range.mean() + 1e-8)
            
            # Volatility Compression Breakout
            recent_volatility = current_data['close'].pct_change().tail(10).std()
            historical_volatility = current_data['close'].pct_change().tail(20).std()
            
            volatility_compression = recent_volatility / (historical_volatility + 1e-8)
        else:
            range_convergence = 1
            volatility_compression = 1
        
        # 4. Overnight Reversal Patterns
        # Close-Open Extreme Return
        if len(current_data) >= 5:
            overnight_returns = []
            for i in range(1, min(6, len(current_data))):
                prev_close = current_data['close'].iloc[-i-1] if -i-1 >= -len(current_data) else current_data['close'].iloc[0]
                current_open = current_data['open'].iloc[-i]
                overnight_ret = (current_open - prev_close) / prev_close
                overnight_returns.append(overnight_ret)
            
            extreme_overnight = np.percentile(overnight_returns, 90) if overnight_returns else 0
            
            # Reversal Intensity Signal
            day_returns = []
            for i in range(min(5, len(current_data))):
                day_ret = (current_data['close'].iloc[-i-1] - current_data['open'].iloc[-i-1]) / current_data['open'].iloc[-i-1] if -i-1 >= -len(current_data) else 0
                day_returns.append(day_ret)
            
            reversal_intensity = 0
            if len(overnight_returns) == len(day_returns):
                reversal_correlation = -np.corrcoef(overnight_returns, day_returns)[0,1] if len(overnight_returns) > 1 else 0
                reversal_intensity = reversal_correlation * np.std(overnight_returns) if not np.isnan(reversal_correlation) else 0
        else:
            extreme_overnight = 0
            reversal_intensity = 0
        
        # Combine factors with weights
        intraday_score = gap_range_ratio * 0.4 + path_efficiency * 0.6
        volume_score = amount_deviation * 0.5 + elasticity * 0.5
        volatility_score = range_convergence * 0.6 + volatility_compression * 0.4
        reversal_score = extreme_overnight * 0.7 + reversal_intensity * 0.3
        
        # Final composite factor
        composite_factor = (
            intraday_score * 0.25 +
            volume_score * 0.25 + 
            volatility_score * 0.25 +
            reversal_score * 0.25
        )
        
        result.loc[date] = composite_factor
    
    return result
