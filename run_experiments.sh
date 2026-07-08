#!/bin/bash
PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/BERTdemo/bin/python
DIR=/Users/DriveD/Documents/JAIST/ContextStudy_NewResearch

cd $DIR

echo "=== base_roberta ===" 
$PYTHON main.py --config configs/base_roberta.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== context_roberta ==="
$PYTHON main.py --config configs/context_roberta.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== retrieval_k3_chr_roberta ==="
$PYTHON main.py --config configs/retrieval_k3_chr_roberta.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== retrieval_k3_chr_xlmr ==="
$PYTHON main.py --config configs/retrieval_k3_chr_xlmr.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== retrieval_k3_asc_roberta ==="
$PYTHON main.py --config configs/retrieval_k3_asc_roberta.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== retrieval_k3_asc_xlmr ==="
$PYTHON main.py --config configs/retrieval_k3_asc_xlmr.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== dual_stream_roberta ==="
$PYTHON main.py --config configs/dual_stream_roberta.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== dual_stream_xlmr ==="
$PYTHON main.py --config configs/dual_stream_xlmr.json --device mps \
  2>&1 | grep -v "Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

echo "=== ALL DONE ==="
