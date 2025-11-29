import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining multiple market microstructure signals
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        current_data = df.loc[:date].copy()
        
        if len(current_data) < 20:  # Minimum data requirement
            result.loc[date] = 0
            continue
            
        # Volatility Regime Detection
        # Multi-timeframe volatility ratios
        vol_5d = current_data['close'].pct_change().rolling(5).std()
        vol_20d = current_data['close'].pct_change().rolling(20).std()
        vol_ratio = vol_5d / vol_20d
        
        # Volume-volatility coupling
        volume_vol_corr = current_data['volume'].rolling(10).corr(current_data['high'] - current_data['low'])
        
        # Momentum Asymmetry
        # Directional price strength
        up_days = (current_data['close'] > current_data['open']).rolling(5).sum()
        total_days = 5
        directional_strength = (up_days - (total_days - up_days)) / total_days
        
        # Volume-momentum divergence
        price_momentum = current_data['close'].pct_change(5)
        volume_momentum = current_data['volume'].pct_change(5)
        mom_divergence = price_momentum - volume_momentum
        
        # Opening Auction Dynamics
        # Auction price pressure
        open_gap = (current_data['open'] - current_data['close'].shift(1)) / current_data['close'].shift(1)
        auction_pressure = open_gap.rolling(3).mean()
        
        # Post-auction validation
        auction_validation = ((current_data['high'] - current_data['open']) / current_data['open']) - \
                           ((current_data['open'] - current_data['low']) / current_data['open'])
        
        # Liquidity Efficiency
        # Price impact per unit volume
        price_range = current_data['high'] - current_data['low']
        volume_impact = price_range / current_data['volume']
        
        # Liquidity absorption rate
        typical_price = (current_data['high'] + current_data['low'] + current_data['close']) / 3
        liquidity_absorption = (current_data['amount'] / current_data['volume']) / typical_price
        
        # Session Phase Analysis
        # Morning momentum establishment (first hour proxy)
        morning_range = (current_data['high'] - current_data['low']) / current_data['open']
        morning_momentum = morning_range.rolling(3).mean()
        
        # Phase transition signals
        close_position = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low'])
        phase_signal = close_position.rolling(5).std()
        
        # Price Rejection Strength
        # Support/resistance testing intensity
        upper_shadow = (current_data['high'] - np.maximum(current_data['close'], current_data['open'])) / current_data['close']
        lower_shadow = (np.minimum(current_data['close'], current_data['open']) - current_data['low']) / current_data['close']
        rejection_strength = (upper_shadow - lower_shadow).abs().rolling(5).mean()
        
        # Post-rejection momentum
        prev_rejection = (upper_shadow.shift(1) > 0.02) | (lower_shadow.shift(1) > 0.02)
        post_rejection_ret = current_data['close'].pct_change()
        rejection_momentum = post_rejection_ret.where(prev_rejection, 0).rolling(3).mean()
        
        # Combine factors with appropriate weights
        current_values = {
            'vol_regime': vol_ratio.iloc[-1] if not pd.isna(vol_ratio.iloc[-1]) else 0,
            'vol_volume_coupling': volume_vol_corr.iloc[-1] if not pd.isna(volume_vol_corr.iloc[-1]) else 0,
            'directional_strength': directional_strength.iloc[-1] if not pd.isna(directional_strength.iloc[-1]) else 0,
            'momentum_divergence': mom_divergence.iloc[-1] if not pd.isna(mom_divergence.iloc[-1]) else 0,
            'auction_pressure': auction_pressure.iloc[-1] if not pd.isna(auction_pressure.iloc[-1]) else 0,
            'auction_validation': auction_validation.iloc[-1] if not pd.isna(auction_validation.iloc[-1]) else 0,
            'price_impact': -volume_impact.iloc[-1] if not pd.isna(volume_impact.iloc[-1]) else 0,
            'liquidity_absorption': liquidity_absorption.iloc[-1] if not pd.isna(liquidity_absorption.iloc[-1]) else 0,
            'morning_momentum': morning_momentum.iloc[-1] if not pd.isna(morning_momentum.iloc[-1]) else 0,
            'phase_signal': -phase_signal.iloc[-1] if not pd.isna(phase_signal.iloc[-1]) else 0,
            'rejection_strength': -rejection_strength.iloc[-1] if not pd.isna(rejection_strength.iloc[-1]) else 0,
            'rejection_momentum': rejection_momentum.iloc[-1] if not pd.isna(rejection_momentum.iloc[-1]) else 0
        }
        
        # Final factor combination (equal weights for demonstration)
        factor_value = sum(current_values.values()) / len([v for v in current_values.values() if v != 0])
        
        result.loc[date] = factor_value if not pd.isna(factor_value) else 0
    
    return result
