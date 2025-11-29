import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Relative Momentum and Liquidity Flow Factor
    Combines momentum quality assessment with liquidity flow dynamics
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    # Minimum required data points
    min_periods = 20
    
    for current_date in data.index[min_periods:]:
        current_idx = data.index.get_loc(current_date)
        if current_idx < min_periods:
            continue
            
        # Get current and historical data (only past information)
        current_data = data.iloc[:current_idx+1]
        
        try:
            # 1. Cross-Asset Relative Price Momentum Components
            # Stock returns
            stock_5d_return = (current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1) if len(current_data) >= 6 else 0
            stock_3d_return = (current_data['close'].iloc[-1] / current_data['close'].iloc[-4] - 1) if len(current_data) >= 4 else 0
            stock_10d_return = (current_data['close'].iloc[-1] / current_data['close'].iloc[-11] - 1) if len(current_data) >= 11 else 0
            
            # Previous period returns for acceleration
            prev_stock_3d_return = (current_data['close'].iloc[-4] / current_data['close'].iloc[-7] - 1) if len(current_data) >= 7 else 0
            
            # Momentum quality measures
            daily_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
            volatility_adjusted_momentum = stock_5d_return / (daily_range / current_data['close'].iloc[-1] + 1e-8) if daily_range > 0 else 0
            
            # Momentum consistency
            return_ratio = (1 + stock_3d_return) / (1 + stock_10d_return + 1e-8)
            momentum_acceleration = stock_3d_return - prev_stock_3d_return
            
            # 2. Liquidity Flow Dynamics Components
            # Price efficiency
            price_efficiency = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / (daily_range + 1e-8)
            volume_weighted_efficiency = price_efficiency * current_data['volume'].iloc[-1]
            
            # Volume analysis
            recent_volume = current_data['volume'].iloc[-10:] if len(current_data) >= 10 else current_data['volume']
            volume_ratio = current_data['volume'].iloc[-1] / (np.mean(recent_volume) + 1e-8)
            
            # Volume trend (using 5-day simple moving average)
            if len(current_data) >= 6:
                volume_5d_ma = current_data['volume'].iloc[-5:].mean()
                volume_prev_5d_ma = current_data['volume'].iloc[-10:-5].mean() if len(current_data) >= 10 else volume_5d_ma
                volume_trend_acceleration = (volume_5d_ma - volume_prev_5d_ma) / (volume_prev_5d_ma + 1e-8)
            else:
                volume_trend_acceleration = 0
            
            # Multi-day efficiency correlation (3-day window)
            if len(current_data) >= 4:
                efficiency_values = []
                for i in range(3):
                    idx = -1 - i
                    if idx >= 0:
                        day_range = current_data['high'].iloc[idx] - current_data['low'].iloc[idx]
                        eff = (current_data['close'].iloc[idx] - current_data['open'].iloc[idx]) / (day_range + 1e-8)
                        efficiency_values.append(eff)
                
                if len(efficiency_values) >= 2:
                    efficiency_persistence = np.corrcoef(range(len(efficiency_values)), efficiency_values)[0,1] if not np.isnan(np.corrcoef(range(len(efficiency_values)), efficiency_values)[0,1]) else 0
                else:
                    efficiency_persistence = 0
            else:
                efficiency_persistence = 0
            
            # 3. Combined Momentum-Liquidity Integration
            # Momentum quality score
            momentum_quality = (
                0.4 * volatility_adjusted_momentum +
                0.3 * return_ratio +
                0.3 * (1 if momentum_acceleration > 0 else -1)
            )
            
            # Liquidity strength score
            liquidity_strength = (
                0.4 * volume_weighted_efficiency / (np.std(current_data['volume'].iloc[-10:]) + 1e-8) +
                0.3 * volume_ratio +
                0.3 * efficiency_persistence
            )
            
            # Cross-asset relative positioning (simplified - using absolute measures)
            relative_strength = (
                0.5 * momentum_quality +
                0.5 * liquidity_strength
            )
            
            # 4. Dynamic Weighting based on volatility regime
            recent_volatility = current_data['close'].pct_change().iloc[-10:].std() if len(current_data) >= 11 else 0.02
            avg_volatility = current_data['close'].pct_change().iloc[-20:].std() if len(current_data) >= 21 else 0.02
            
            # Volatility regime detection
            high_vol_regime = recent_volatility > avg_volatility * 1.2
            
            if high_vol_regime:
                # Emphasize liquidity signals in high volatility
                final_signal = (
                    0.3 * momentum_quality +
                    0.7 * liquidity_strength
                )
            else:
                # Emphasize momentum signals in low volatility
                final_signal = (
                    0.7 * momentum_quality +
                    0.3 * liquidity_strength
                )
            
            # Apply momentum-liquidity convergence logic
            momentum_strength = 1 if momentum_quality > 0 else -1
            liquidity_strength_dir = 1 if liquidity_strength > 0 else -1
            
            if momentum_strength == 1 and liquidity_strength_dir == 1:
                # Strong momentum + Strong liquidity - bullish convergence
                convergence_multiplier = 1.2
            elif momentum_strength == -1 and liquidity_strength_dir == -1:
                # Weak momentum + Weak liquidity - bearish convergence
                convergence_multiplier = -1.2
            elif momentum_strength == 1 and liquidity_strength_dir == -1:
                # Strong momentum + Weak liquidity - caution signal
                convergence_multiplier = 0.5
            elif momentum_strength == -1 and liquidity_strength_dir == 1:
                # Weak momentum + Strong liquidity - reversal potential
                convergence_multiplier = -0.5
            else:
                convergence_multiplier = 1.0
            
            final_signal *= convergence_multiplier
            
            alpha_signal.loc[current_date] = final_signal
            
        except (IndexError, ZeroDivisionError, ValueError):
            alpha_signal.loc[current_date] = 0
    
    # Fill initial NaN values with 0
    alpha_signal = alpha_signal.fillna(0)
    
    return alpha_signal
