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

echo "=== ALL DONE ==="
