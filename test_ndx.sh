#!/bin/bash

# cd Treevo
# conda activate ReEvo
# bash test_ndx.sh

set -a
source ~/.env
set +a

python main.py \
    problem=ndx \
    init_pop_size=10 \
    pop_size=10 \
    max_fe=200 \
    object_n=5 \
    timeout=10 \
    algorithm=TreEvo \
    llm_client=openrouter \