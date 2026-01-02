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
    timeout=10 \
    algorithm=eoh \
    llm_client=openrouter \
    llm_client.api_key='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'