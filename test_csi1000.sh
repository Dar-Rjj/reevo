#!/bin/bash

python main.py \
    problem=csi1000 \
    init_pop_size=4 \
    pop_size=4 \
    max_fe=20 \
    timeout=20 \
    algorithm=treevoo \
    llm_client=qwen3_api \