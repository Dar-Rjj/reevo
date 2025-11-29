#!/bin/bash

# cd Treevo
# conda activate ReEvo
# sh test_csi300.sh

python main.py \
    problem=csi300 \
    init_pop_size=10 \
    pop_size=10 \
    max_fe=200 \
    object_n=5 \
    timeout=10 \
    algorithm=TReEvo \
    llm_client=deepseek \
    llm_client.api_key='sk-be2718f6b577422c82011a304204a0a8'