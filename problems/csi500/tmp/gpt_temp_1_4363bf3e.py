import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining multiple technical components:
    - Volatility-Weighted Gap Momentum with Efficiency
    - Range Breakout Quality with Pressure Asymmetry  
    - Intraday Range-Pressure Momentum with Divergence
    - Efficiency-Weighted Compression Breakout
    - Momentum Diffusion with Range-Volume Divergence
    """
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Volatility-Weighted Gap Momentum with Efficiency
    def calculate_gap_momentum(data):
        # Overnight gap
        overnight_gap = (data['open'] / data['close'].shift(1) - 1)
        
        # Gap efficiency (avoid division by zero)
        gap_efficiency = data['amount'] / (data['open'] * data['volume'].replace(0, np.nan))
        gap_efficiency = gap_efficiency.fillna(0)
        
        # Volatility asymmetry
        upside_vol = (data['high'] - data['open']) / data['open'].replace(0, np.nan)
        downside_vol = (data['open'] - data['low']) / data['open'].replace(0, np.nan)
        vol_divergence = upside_vol - downside_vol
        
        # Persistence logic for consecutive gaps
        gap_sign = np.sign(overnight_gap)
        persistence = gap_sign.groupby(gap_sign.index).transform(
            lambda x: x.eq(x.shift(1)) & x.eq(x.shift(2))
        ).astype(int) + 1
        
        # Combined gap momentum signal
        gap_momentum = overnight_gap * vol_divergence * gap_efficiency * persistence
        return gap_momentum
    
    # Range Breakout Quality with Pressure Asymmetry
    def calculate_breakout_quality(data):
        # Daily price range
        daily_range = data['high'] - data['low']
        range_ratio = daily_range / daily_range.rolling(5).mean()
        
        # Pressure asymmetry
        buying_pressure = (data['close'] - data['low']) / daily_range.replace(0, np.nan)
        selling_pressure = (data['high'] - data['close']) / daily_range.replace(0, np.nan)
        pressure_asymmetry = buying_pressure - selling_pressure
        
        # Price discovery efficiency
        price_efficiency = (data['open'] - data['close'].shift(1)).abs() / daily_range.replace(0, np.nan)
        price_efficiency = price_efficiency.fillna(0)
        
        # Volume confirmation
        volume_ratio = data['volume'] / data['volume'].rolling(5).mean()
        volume_asymmetry = (data['volume'] - data['volume'].shift(1)) / data['volume'].rolling(5).std().replace(0, np.nan)
        volume_asymmetry = volume_asymmetry.fillna(0)
        
        # Enhanced breakout signal
        breakout_signal = range_ratio * pressure_asymmetry * price_efficiency * volume_ratio * (1 + volume_asymmetry)
        return breakout_signal
    
    # Intraday Range-Pressure Momentum with Divergence
    def calculate_intraday_momentum(data):
        # Net range pressure
        daily_range = data['high'] - data['low']
        buying_pressure = (data['close'] - data['low']) / daily_range.replace(0, np.nan)
        selling_pressure = (data['high'] - data['close']) / daily_range.replace(0, np.nan)
        net_pressure = buying_pressure - selling_pressure
        
        # True range persistence
        true_range = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        tr_persistence = true_range / true_range.rolling(3).mean()
        
        # Momentum divergence
        price_momentum = data['close'].pct_change(3)
        vol_momentum = true_range.pct_change(3)
        
        # Pressure-price divergence
        pressure_trend = net_pressure.rolling(3).mean()
        price_trend = data['close'].pct_change(3)
        
        positive_divergence = ((pressure_trend > pressure_trend.shift(1)) & 
                              (price_trend < price_trend.shift(1))).astype(int)
        negative_divergence = ((pressure_trend > pressure_trend.shift(1)) & 
                              (price_trend > price_trend.shift(1))).astype(int)
        
        divergence_strength = positive_divergence - negative_divergence
        
        # Volume trend confirmation
        volume_trend = data['volume'].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        )
        volume_direction = np.sign(volume_trend)
        
        # Combined intraday momentum
        intraday_momentum = net_pressure * divergence_strength * tr_persistence * volume_direction
        return intraday_momentum
    
    # Efficiency-Weighted Compression Breakout
    def calculate_compression_breakout(data):
        # Price compression
        price_compression = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2).replace(0, np.nan)
        compression_periods = price_compression < price_compression.rolling(10).median()
        
        # Trading efficiency
        trading_efficiency = data['amount'] / (data['close'] * data['volume'].replace(0, np.nan))
        trading_efficiency = trading_efficiency.fillna(0)
        
        # Volume-confirmed breakouts
        volume_ratio = data['volume'] / data['volume'].rolling(5).mean()
        volume_surge = (volume_ratio > 1.2) & compression_periods
        
        # Directional component
        intraday_direction = np.sign(data['close'] - data['open'])
        directional_momentum = data['close'].pct_change(3)
        
        # Smart breakout signal
        compression_signal = (1 / (price_compression + 1e-6)) * volume_ratio * trading_efficiency * intraday_direction * (1 + directional_momentum)
        compression_signal = compression_signal * volume_surge.astype(int)
        return compression_signal
    
    # Momentum Diffusion with Range-Volume Divergence
    def calculate_momentum_diffusion(data):
        # Enhanced momentum components
        core_momentum = (data['close'] - data['open']) / data['open'].replace(0, np.nan)
        
        # Momentum spread using intraday distribution
        high_momentum = (data['high'] - data['open']) / data['open'].replace(0, np.nan)
        low_momentum = (data['open'] - data['low']) / data['open'].replace(0, np.nan)
        momentum_spread = high_momentum - low_momentum
        
        # Volume momentum
        volume_momentum = data['volume'].rolling(3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 3 else 0
        )
        
        # Range-volume divergence
        true_range = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        range_direction = true_range.pct_change(3)
        volume_direction = volume_momentum.pct_change(3)
        
        # Divergence patterns
        positive_divergence = ((range_direction > 0) & (volume_direction > 0)).astype(int)
        negative_divergence = ((range_direction < 0) & (volume_direction < 0)).astype(int)
        divergence_strength = positive_divergence - negative_divergence
        
        # Quality assessment
        volume_support = (data['volume'] > data['volume'].rolling(5).mean()).astype(int)
        price_consistency = (data['close'].rolling(3).std() / data['close'].rolling(3).mean()).replace(0, np.nan)
        price_consistency = (1 / price_consistency).fillna(0)
        
        # Composite diffusion factor
        diffusion_factor = momentum_spread * divergence_strength * volume_support * price_consistency
        return diffusion_factor
    
    # Calculate all components
    gap_momentum = calculate_gap_momentum(data)
    breakout_quality = calculate_breakout_quality(data)
    intraday_momentum = calculate_intraday_momentum(data)
    compression_breakout = calculate_compression_breakout(data)
    momentum_diffusion = calculate_momentum_diffusion(data)
    
    # Combine all components with equal weighting
    combined_factor = (
        gap_momentum.fillna(0) +
        breakout_quality.fillna(0) +
        intraday_momentum.fillna(0) +
        compression_breakout.fillna(0) +
        momentum_diffusion.fillna(0)
    ) / 5
    
    # Normalize the final factor
    result = (combined_factor - combined_factor.rolling(20).mean()) / combined_factor.rolling(20).std()
    result = result.fillna(0)
    
    return result
