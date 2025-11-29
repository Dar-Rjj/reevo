import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Range-Pressure Efficiency with Volume-Confirmation factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Required periods for calculations
    short_period = 5
    medium_period = 10
    long_period = 20
    
    for current_date in df.index:
        current_idx = df.index.get_loc(current_date)
        
        # Skip if insufficient data
        if current_idx < long_period:
            factor_values.loc[current_date] = np.nan
            continue
        
        # Get current and historical data
        current_data = df.iloc[current_idx]
        hist_data = df.iloc[:current_idx+1]  # Include current day
        
        # 1. Opening-Closing Pressure Dynamics
        # Multi-Period Gap Pressure Analysis
        if current_idx >= 1:
            prev_close = df.iloc[current_idx-1]['close']
            current_open = current_data['open']
            current_close = current_data['close']
            current_high = current_data['high']
            current_low = current_data['low']
            
            # Opening gap pressure
            gap_pressure = (current_open - prev_close) / prev_close
            
            # Closing pressure persistence
            close_pressure = (current_close - current_open) / current_open
            
            # Range-constrained pressure efficiency
            daily_range = (current_high - current_low) / current_low
            upper_pressure = (current_high - current_open) / current_open
            lower_pressure = (current_open - current_low) / current_open
            
            range_efficiency = upper_pressure / (daily_range + 1e-8) if daily_range > 0 else 0
            
        else:
            gap_pressure = close_pressure = range_efficiency = 0
        
        # Multi-period calculations
        if current_idx >= medium_period:
            recent_data = hist_data.iloc[-medium_period:]
            
            # Gap pressure momentum
            gap_momentum = []
            for i in range(1, len(recent_data)):
                if i > 0:
                    gap = (recent_data.iloc[i]['open'] - recent_data.iloc[i-1]['close']) / recent_data.iloc[i-1]['close']
                    gap_momentum.append(gap)
            
            gap_momentum_strength = np.mean(gap_momentum) if gap_momentum else 0
            
            # Closing pressure persistence
            close_pressures = []
            for i in range(len(recent_data)):
                if i > 0:
                    pressure = (recent_data.iloc[i]['close'] - recent_data.iloc[i]['open']) / recent_data.iloc[i]['open']
                    close_pressures.append(pressure)
            
            close_persistence = np.std(close_pressures) if close_pressures else 0
            
        else:
            gap_momentum_strength = close_persistence = 0
        
        # 2. Volume-Pressure Equilibrium Analysis
        current_volume = current_data['volume']
        current_amount = current_data['amount']
        
        if current_idx >= short_period:
            volume_data = hist_data['volume'].iloc[-short_period:]
            amount_data = hist_data['amount'].iloc[-short_period:]
            
            # Volume momentum and concentration
            volume_momentum = current_volume / volume_data.mean() if volume_data.mean() > 0 else 1
            volume_concentration = current_volume / (volume_data.std() + 1e-8)
            
            # Volume-pressure alignment
            if current_idx >= 1:
                price_change = (current_data['close'] - hist_data.iloc[current_idx-1]['close']) / hist_data.iloc[current_idx-1]['close']
                volume_pressure_alignment = abs(price_change) * volume_momentum
            else:
                volume_pressure_alignment = 0
            
            # Amount-driven inefficiency
            vwap = current_amount / current_volume if current_volume > 0 else current_data['close']
            price_vwap_deviation = (current_data['close'] - vwap) / vwap
            amount_efficiency = abs(price_vwap_deviation) * (current_amount / amount_data.mean() if amount_data.mean() > 0 else 1)
            
        else:
            volume_momentum = volume_concentration = volume_pressure_alignment = amount_efficiency = 0
        
        # 3. Volatility-Regime Transition Efficiency
        if current_idx >= long_period:
            price_data = hist_data['close'].iloc[-long_period:]
            returns = price_data.pct_change().dropna()
            
            # Multi-timeframe volatility
            short_vol = returns.iloc[-short_period:].std() if len(returns) >= short_period else 0
            medium_vol = returns.iloc[-medium_period:].std() if len(returns) >= medium_period else 0
            long_vol = returns.std()
            
            # Volatility regime identification
            volatility_ratio = short_vol / (long_vol + 1e-8)
            regime_transition = abs(volatility_ratio - 1)
            
            # Pressure efficiency in volatility context
            if short_vol > 0:
                pressure_efficiency_vol = abs(close_pressure) / short_vol
            else:
                pressure_efficiency_vol = 0
                
        else:
            volatility_ratio = regime_transition = pressure_efficiency_vol = 0
        
        # 4. Multi-Dimensional Efficiency Signal Generation
        # Combine all components with appropriate weights
        pressure_component = 0.3 * gap_pressure + 0.4 * close_pressure + 0.3 * range_efficiency
        volume_component = 0.4 * volume_pressure_alignment + 0.3 * volume_concentration + 0.3 * amount_efficiency
        volatility_component = 0.5 * pressure_efficiency_vol + 0.3 * regime_transition + 0.2 * volatility_ratio
        
        # Final factor value with cross-dimensional validation
        momentum_validation = gap_momentum_strength * close_persistence if gap_momentum_strength * close_persistence > 0 else 0
        efficiency_coherence = (pressure_component * volume_component * volatility_component) ** (1/3)
        
        final_factor = (
            0.4 * pressure_component +
            0.35 * volume_component +
            0.25 * volatility_component +
            0.1 * efficiency_coherence +
            0.05 * momentum_validation
        )
        
        factor_values.loc[current_date] = final_factor
    
    # Normalize the factor values
    if not factor_values.isna().all():
        factor_values = (factor_values - factor_values.mean()) / (factor_values.std() + 1e-8)
    
    return factor_values
