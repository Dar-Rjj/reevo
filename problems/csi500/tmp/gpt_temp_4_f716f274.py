import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Fragmentation with Price-Level Anchoring factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(5, len(data)):
        current_date = data.index[i]
        current_data = data.iloc[i]
        historical_data = data.iloc[:i+1]  # Only past and current data
        
        # Skip if insufficient data
        if len(historical_data) < 6:
            factor.iloc[i] = 0
            continue
            
        try:
            # MOMENTUM FRAGMENTATION COMPONENT
            # Use available data to approximate session behavior
            # For simplicity, we'll use rolling windows to approximate intraday patterns
            
            # Morning session approximation (first 25% of trading hours)
            morning_window = max(1, len(historical_data) // 4)
            morning_data = historical_data.iloc[-morning_window:]
            
            morning_return = (morning_data['high'].max() - morning_data['open'].iloc[0]) / morning_data['open'].iloc[0]
            morning_volatility = (morning_data['high'].max() - morning_data['low'].min()) / morning_data['open'].iloc[0]
            morning_volume_intensity = morning_data['volume'].sum() / (historical_data['volume'].sum() / 6.5)
            
            # Midday session approximation (middle 50%)
            midday_window = max(1, len(historical_data) // 2)
            midday_start = max(0, len(historical_data) - midday_window - morning_window)
            midday_data = historical_data.iloc[midday_start:midday_start + midday_window]
            
            if len(midday_data) > 1:
                midday_direction = np.sign(midday_data['close'].iloc[-1] - midday_data['high'].iloc[0])
                midday_range_compression = ((midday_data['high'].max() - midday_data['low'].min()) / 
                                          (morning_data['high'].max() - morning_data['low'].min()))
                midday_volume_decay = midday_data['volume'].sum() / morning_data['volume'].sum()
            else:
                midday_direction = 0
                midday_range_compression = 1
                midday_volume_decay = 1
            
            # Afternoon session approximation (last 25%)
            afternoon_data = historical_data.iloc[-min(morning_window, len(historical_data)):]
            
            if len(afternoon_data) > 1 and len(midday_data) > 1:
                afternoon_momentum = (afternoon_data['close'].iloc[-1] - afternoon_data['open'].iloc[0]) / afternoon_data['open'].iloc[0]
                afternoon_range_expansion = ((afternoon_data['high'].max() - afternoon_data['low'].min()) / 
                                           (midday_data['high'].max() - midday_data['low'].min()))
                late_volume_surge = afternoon_data['volume'].sum() / midday_data['volume'].sum()
            else:
                afternoon_momentum = 0
                afternoon_range_expansion = 1
                late_volume_surge = 1
            
            # Momentum Fragmentation Index
            session_returns = [morning_return, midday_direction, afternoon_momentum]
            session_return_dispersion = np.std(session_returns) if len(session_returns) > 1 else 0
            
            volatility_regime_changes = abs(morning_volatility - afternoon_range_expansion)
            
            # PRICE-LEVEL ANCHORING COMPONENT
            # Historical price anchors
            prev_day_high = historical_data['high'].iloc[-2] if len(historical_data) > 1 else current_data['high']
            prev_day_low = historical_data['low'].iloc[-2] if len(historical_data) > 1 else current_data['low']
            prev_day_close = historical_data['close'].iloc[-2] if len(historical_data) > 1 else current_data['close']
            
            # Weekly anchors (5-day window)
            weekly_window = min(5, len(historical_data))
            weekly_high = historical_data['high'].iloc[-weekly_window:].max()
            weekly_low = historical_data['low'].iloc[-weekly_window:].min()
            weekly_open = historical_data['open'].iloc[-weekly_window]
            
            # Price-anchor distance effects
            proximity_to_high = (current_data['close'] - weekly_high) / weekly_high
            distance_from_low = (current_data['close'] - weekly_low) / weekly_low
            prev_close_distance = (current_data['close'] - prev_day_close) / prev_day_close
            
            # Anchor interaction dynamics
            high_anchor_strength = abs(proximity_to_high) < 0.02  # Within 2% of weekly high
            low_anchor_strength = abs(distance_from_low) < 0.02   # Within 2% of weekly low
            
            # FRAGMENTATION-ANCHORING SYNTHESIS
            # Session-specific anchor effects
            morning_anchor_effect = morning_return * (1 if high_anchor_strength else -1 if low_anchor_strength else 0)
            afternoon_anchor_effect = afternoon_momentum * (1 if high_anchor_strength else -1 if low_anchor_strength else 0)
            
            # Momentum fragmentation around anchors
            pre_anchor_momentum = morning_return if high_anchor_strength or low_anchor_strength else 0
            anchor_test_momentum = midday_direction if high_anchor_strength or low_anchor_strength else 0
            
            # ALPHA SIGNAL GENERATION
            # Fragmentation-Anchoring Score
            session_consistency_score = 1 - session_return_dispersion
            anchor_proximity_impact = -abs(proximity_to_high) - abs(distance_from_low)  # Negative when near extremes
            
            # Multi-timeframe signal integration
            intraday_momentum = (morning_return + afternoon_momentum) / 2
            anchor_significance = 1 if high_anchor_strength or low_anchor_strength else 0
            
            # Final alpha construction
            fragmentation_component = (session_consistency_score * intraday_momentum * 
                                     (1 + volatility_regime_changes))
            
            anchoring_component = (anchor_proximity_impact * anchor_significance * 
                                 (morning_anchor_effect + afternoon_anchor_effect))
            
            raw_factor = fragmentation_component + anchoring_component
            
            # Session-specific adjustments
            volume_adjustment = np.log1p(morning_volume_intensity + late_volume_surge)
            volatility_adjustment = 1 / (1 + morning_volatility)
            
            final_factor = raw_factor * volume_adjustment * volatility_adjustment
            
            factor.iloc[i] = final_factor
            
        except (ValueError, ZeroDivisionError, IndexError):
            factor.iloc[i] = 0
    
    # Fill any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
