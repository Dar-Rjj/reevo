#!/bin/bash

# cd Treevo
# conda activate ReEvo
# sh test_csi500.sh

python main.py \
    problem=csi500 \
    init_pop_size=10 \
    pop_size=10 \
    max_fe=200 \
    object_n=5 \
    timeout=20 \
    algorithm=TreEvo \
    llm_client=deepseek \
    llm_client.api_key='sk-be2718f6b577422c82011a304204a0a8'