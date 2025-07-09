import numpy as np
import pandas as pd
import os
import sys
import logging
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)

def solve(market_data):
    factor = heuristics(market_data)
    

# 用法示例
if __name__ == "__main__":
    print(f"开始回测: {heuristic_name} ...")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 读取CSI300股票列表
    try:
        csi300_path = os.path.join(script_dir, "csi300.txt")
        if os.path.exists(csi300_path):
            print("找到csi300.txt文件")
        else:
            raise FileNotFoundError("无法找到csi300.txt文件。请确保文件位于正确位置。")
    except Exception as e:
        print(f"读取CSI300股票列表时出错: {e}")

    with open(csi300_path, 'r') as f:
        csi300_stocks = [line.strip() for line in f.readlines()]
    
    print(f"CSI300股票列表中共有{len(csi300_stocks)}只股票")

    # 2. 设定日期范围
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2024-01-01')