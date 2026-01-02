import numpy as np
import pandas as pd
import os
import sys
import logging
import importlib.util
sys.path.insert(0, "../../../")

from scipy.stats import pearsonr
from utils.utils import get_heuristic_name


def load_heuristic_func(code_path):
    """
    动态加载临时生成的个体代码文件(如 gpt_temp_xxxx.py), 并获取启发式函数名。
    """
    spec = importlib.util.spec_from_file_location("gpt_module", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]
    heuristic_name = get_heuristic_name(module, possible_func_names)
    heuristics = getattr(module, heuristic_name)
    return heuristics

def solve(market_data, heuristics, mood, object_n):
    # 计算每只股票的因子值
    market_data['factor'] = market_data.groupby('stock_code').apply(lambda x: heuristics(x.droplevel('stock_code')))

    # 计算未来6日收益率
    market_data['future_return'] = market_data.groupby('stock_code')['close'].shift(-object_n) / market_data['close'] - 1

    # 取所有日期
    if mood == 'train':
        start_date, end_date = pd.Timestamp('2016-01-01'), pd.Timestamp('2020-01-01')
    elif mood == 'val':
        start_date, end_date = pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')
    else:
        start_date, end_date = pd.Timestamp('2021-01-01'), pd.Timestamp('2024-01-01')
    
    all_dates = market_data.index.get_level_values('date').unique()
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    ic_values = []

    for date in all_dates:
        daily = market_data.xs(date, level='date')
        factors = daily['factor']
        returns = daily['future_return']
        eps = (np.random.rand(len(factors)) * 2 - 1) * 1e-7
        factors += eps  # 添加微小扰动以避免完全相等

        mask = factors.notna() & returns.notna() & np.isfinite(factors) & np.isfinite(returns)
        if mask.sum() >= 10:
            ic, _ = pearsonr(factors[mask], returns[mask])
            if not np.isnan(ic):
                ic_values.append(ic)

    return np.mean(ic_values) if ic_values else 0
    

if __name__ == "__main__":
    print("[*] Running ...")
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    code_path = sys.argv[4]
    object_n = int(sys.argv[5])
    assert mood in ['train', 'val', 'test']
    basepath = os.path.dirname(__file__)
    dataset_path = os.path.join(basepath, f"{mood}_data.csv")

    market_data = pd.read_csv(dataset_path, parse_dates=['date'])
    market_data.set_index(['stock_code', 'date'], inplace=True)
    market_data.sort_index(inplace=True)

    try:
        heuristics = load_heuristic_func(code_path)
        mean_ic = solve(market_data, heuristics, mood, object_n)
    except Exception as e:
        mean_ic = 0
        logging.error(f"Error in evaluating heuristics from {code_path}: {e}")

    os.remove(code_path)
    print("[*] Average:")
    print(abs(mean_ic))