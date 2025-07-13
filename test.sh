#!/bin/bash

python main.py \
    problem=backtest \
    init_pop_size=4 \
    pop_size=4 \
    max_fe=20 \
    timeout=20 \
    algorithm=treevo \
    llm_client=qwen3_api \
    llm_client.api_key='sk-fb4917a77b7d4a2b88369204d7435aba'