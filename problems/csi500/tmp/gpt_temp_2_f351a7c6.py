import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize factor series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate necessary columns
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['daily_range'] = (df['high'] - df['low']) / df['prev_close']
    df['close_to_open'] = (df['close'] - df['open']) / df['open']
    df['close_to_prev'] = (df['close'] - df['prev_close']) / df['prev_close']
    
    # Calculate volatility for gap normalization
    df['volatility_5d'] = df['close'].pct_change().rolling(window=5).std()
    
    # Price position within daily range
    df['high_proximity'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    df['low_proximity'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume and amount calculations
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['amount_ratio'] = df['amount'] / df['amount'].rolling(window=5).mean()
    
    # Calculate average trade size
    df['avg_trade_size'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['avg_trade_size_ratio'] = df['avg_trade_size'] / df['avg_trade_size'].rolling(window=5).mean()
    
    for date in df.index:
        if pd.isna(df.loc[date, 'prev_close']):
            continue
            
        current_data = df.loc[date]
        
        # Gap Opening Reversal Analysis
        gap_magnitude = current_data['gap']
        vol_adjusted_gap = gap_magnitude / (current_data['volatility_5d'] + 1e-6)
        
        # Gap fill completion
        gap_fill = 0
        if gap_magnitude > 0:  # Gap up
            if current_data['close'] < current_data['prev_close']:
                gap_fill = -1  # Filled gap down
            elif current_data['close'] < current_data['open']:
                gap_fill = -0.5  # Partial fill
        elif gap_magnitude < 0:  # Gap down
            if current_data['close'] > current_data['prev_close']:
                gap_fill = 1  # Filled gap up
            elif current_data['close'] > current_data['open']:
                gap_fill = 0.5  # Partial fill
        
        # Price Extreme Formation
        failed_breakout = 0
        # New high with weak close
        if current_data['high'] == df.loc[:date, 'high'].max():
            if current_data['high_proximity'] > 0.7:  # Close near low of day
                failed_breakout = -1
        # New low with strong close
        if current_data['low'] == df.loc[:date, 'low'].min():
            if current_data['low_proximity'] > 0.7:  # Close near high of day
                failed_breakout = 1
        
        # Volume Concentration Analysis
        volume_signal = 0
        if current_data['volume_ratio'] > 1.5:  # High volume day
            if current_data['high_proximity'] < 0.3:  # Volume at highs
                volume_signal = -1
            elif current_data['low_proximity'] < 0.3:  # Volume at lows
                volume_signal = 1
        
        # Amount-Based Liquidity Signals
        liquidity_signal = 0
        if current_data['amount_ratio'] > 1.5:  # High amount day
            if current_data['avg_trade_size_ratio'] > 1.2:  # Large trades
                if current_data['close_to_open'] < -0.01:  # Down day with big trades
                    liquidity_signal = 1  # Potential support
                elif current_data['close_to_open'] > 0.01:  # Up day with big trades
                    liquidity_signal = -1  # Potential distribution
        
        # Integrated Reversal-Liquidity Factors
        reversal_strength = 0
        
        # Gap reversal with volume confirmation
        if abs(gap_magnitude) > 0.02:  # Significant gap
            if gap_fill != 0 and volume_signal * gap_fill > 0:
                reversal_strength += gap_fill * 2
        
        # Failed breakout with liquidity confirmation
        if failed_breakout != 0:
            if liquidity_signal * failed_breakout > 0:
                reversal_strength += failed_breakout * 1.5
        
        # Multi-dimensional confirmation
        signal_alignment = 0
        signals = [gap_fill, failed_breakout, volume_signal, liquidity_signal]
        positive_signals = sum(1 for s in signals if s > 0)
        negative_signals = sum(1 for s in signals if s < 0)
        
        if positive_signals >= 2:
            signal_alignment = 1
        elif negative_signals >= 2:
            signal_alignment = -1
        
        # Final factor integration
        base_reversal = gap_fill + failed_breakout
        volume_modulated = base_reversal * (1 + 0.5 * abs(volume_signal))
        liquidity_modulated = volume_modulated * (1 + 0.3 * abs(liquidity_signal))
        
        final_factor = liquidity_modulated + signal_alignment + reversal_strength
        
        factor.loc[date] = final_factor
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20, min_periods=1).mean()) / factor.rolling(window=20, min_periods=1).std()
    
    return factor
