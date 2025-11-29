import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic lagged values
    data['Close_prev'] = data['close'].shift(1)
    data['Close_prev2'] = data['close'].shift(2)
    data['Close_prev3'] = data['close'].shift(3)
    data['Close_prev5'] = data['close'].shift(5)
    data['High_prev'] = data['high'].shift(1)
    data['High_prev2'] = data['high'].shift(2)
    data['Low_prev'] = data['low'].shift(1)
    data['Low_prev2'] = data['low'].shift(2)
    data['Open_prev'] = data['open'].shift(1)
    data['Amount_prev'] = data['amount'].shift(1)
    data['Amount_prev3'] = data['amount'].shift(3)
    data['Amount_prev5'] = data['amount'].shift(5)
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate all components
    for i in range(2, len(data)):
        if (pd.notna(data.iloc[i]['Close_prev']) and 
            pd.notna(data.iloc[i]['Close_prev2']) and 
            pd.notna(data.iloc[i]['High_prev']) and 
            pd.notna(data.iloc[i]['Low_prev']) and 
            pd.notna(data.iloc[i]['Amount_prev'])):
            
            # 1. Intraday Range-Amount Divergence Factor
            range_expansion = (data.iloc[i]['high'] - data.iloc[i]['low']) - (data.iloc[i]['High_prev'] - data.iloc[i]['Low_prev'])
            price_momentum = data.iloc[i]['close'] / data.iloc[i]['Close_prev'] - 1
            range_momentum_divergence = range_expansion * price_momentum
            
            raw_amount_flow = data.iloc[i]['amount'] * np.sign(data.iloc[i]['close'] - data.iloc[i]['Close_prev'])
            prev_raw_amount_flow = data.iloc[i]['Amount_prev'] * np.sign(data.iloc[i]['Close_prev'] - data.iloc[i]['Close_prev2'])
            amount_flow_change = raw_amount_flow - prev_raw_amount_flow
            intraday_range_amount = range_momentum_divergence * amount_flow_change
            
            # 2. Shadow-Amount Pressure Efficiency
            upper_shadow_pressure = (data.iloc[i]['high'] - max(data.iloc[i]['open'], data.iloc[i]['close'])) - (data.iloc[i]['High_prev'] - max(data.iloc[i]['Open_prev'], data.iloc[i]['Close_prev']))
            lower_shadow_support = (min(data.iloc[i]['open'], data.iloc[i]['close']) - data.iloc[i]['low']) - (min(data.iloc[i]['Open_prev'], data.iloc[i]['Close_prev']) - data.iloc[i]['Low_prev'])
            net_shadow_pressure = upper_shadow_pressure - lower_shadow_support
            
            amount_flow_intensity = data.iloc[i]['amount'] / data.iloc[i]['Amount_prev'] if data.iloc[i]['Amount_prev'] != 0 else 1
            shadow_amount_efficiency = net_shadow_pressure * amount_flow_intensity
            
            # 3. Multi-Timeframe Compression-Amount Alignment
            short_term_compression = (data.iloc[i]['high'] - data.iloc[i]['low']) / ((data.iloc[i]['High_prev'] - data.iloc[i]['Low_prev']) + (data.iloc[i]['High_prev2'] - data.iloc[i]['Low_prev2'])) * 2 if (data.iloc[i]['High_prev'] - data.iloc[i]['Low_prev'] + data.iloc[i]['High_prev2'] - data.iloc[i]['Low_prev2']) != 0 else 0
            medium_term_trend = (data.iloc[i]['close'] - data.iloc[i]['Close_prev5']) / data.iloc[i]['Close_prev5'] if data.iloc[i]['Close_prev5'] != 0 else 0
            compression_trend_divergence = short_term_compression * medium_term_trend
            
            amount_flow_trend = data.iloc[i]['amount'] / data.iloc[i]['Amount_prev5'] if data.iloc[i]['Amount_prev5'] != 0 else 1
            compression_amount_alignment = compression_trend_divergence * amount_flow_trend
            
            # 4. Price-Amount Trend Consistency
            price_trend_strength = (data.iloc[i]['close'] / data.iloc[i]['Close_prev3']) * (data.iloc[i]['close'] / data.iloc[i]['Close_prev5']) if data.iloc[i]['Close_prev3'] != 0 and data.iloc[i]['Close_prev5'] != 0 else 1
            amount_trend_strength = (data.iloc[i]['amount'] / data.iloc[i]['Amount_prev3']) * (data.iloc[i]['amount'] / data.iloc[i]['Amount_prev5']) if data.iloc[i]['Amount_prev3'] != 0 and data.iloc[i]['Amount_prev5'] != 0 else 1
            
            price_amount_correlation = price_trend_strength * amount_trend_strength
            trend_consistency = np.sign(price_trend_strength) * np.sign(amount_trend_strength)
            price_amount_consistency = price_amount_correlation * trend_consistency
            
            # 5. Intraday Range-Amount Momentum Composite
            intraday_range_momentum = (data.iloc[i]['high'] - data.iloc[i]['low']) / data.iloc[i]['low'] if data.iloc[i]['low'] != 0 else 0
            daily_return = data.iloc[i]['close'] / data.iloc[i]['Close_prev'] - 1
            amount_flow_momentum = data.iloc[i]['amount'] / data.iloc[i]['Amount_prev'] - 1
            
            range_return_score = intraday_range_momentum * daily_return
            range_amount_momentum = range_return_score * amount_flow_momentum
            
            # 6. Range-Volatility-Amount Alignment
            range_volatility = (data.iloc[i]['high'] - data.iloc[i]['low']) / data.iloc[i]['close'] if data.iloc[i]['close'] != 0 else 0
            prev_range_volatility = (data.iloc[i]['High_prev'] - data.iloc[i]['Low_prev']) / data.iloc[i]['Close_prev'] if data.iloc[i]['Close_prev'] != 0 else 0
            volatility_change = range_volatility - prev_range_volatility
            
            amount_flow_direction = np.sign(data.iloc[i]['amount'] - data.iloc[i]['Amount_prev'])
            volatility_amount_alignment = volatility_change * amount_flow_direction
            momentum_enhanced_signal = price_momentum * volatility_amount_alignment
            
            # Combine all factors with equal weights
            combined_factor = (
                intraday_range_amount + 
                shadow_amount_efficiency + 
                compression_amount_alignment + 
                price_amount_consistency + 
                range_amount_momentum + 
                momentum_enhanced_signal
            ) / 6.0
            
            factor.iloc[i] = combined_factor
    
    # Fill NaN values with 0
    factor = factor.fillna(0)
    
    return factor
