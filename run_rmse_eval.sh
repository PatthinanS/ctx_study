#!/bin/bash
PYTHON=.venv/bin/python
DIR=/Users/DriveD/Documents/JAIST/ContextStudy_NewResearch
FILTER="Loading weights\|\[transformers\]\|Key \|----\|lm_head\|pooler\|Notes\|UNEXPECTED\|MISSING\|newly init"

cd $DIR

run() { $PYTHON eval_rmse.py --config "$1" 2>&1 | grep -v "$FILTER"; }

# C0 — no context
echo "=== base_roberta ===";        run configs/roberta/base/base_roberta.json
echo "=== base_xlmr ===";           run configs/xlmr/base/base_xlmr.json

# C1 — window
echo "=== context_roberta ===";     run configs/roberta/window/context_roberta.json
echo "=== context_xlmr ===";        run configs/xlmr/window/context_xlmr.json

# C2 — retrieval (RoBERTa)
echo "=== retrieval_k1_roberta ===";       run configs/roberta/retrieval/retrieval_k1_roberta.json
echo "=== retrieval_k2_chr_roberta ===";   run configs/roberta/retrieval/retrieval_k2_chr_roberta.json
echo "=== retrieval_k2_asc_roberta ===";   run configs/roberta/retrieval/retrieval_k2_asc_roberta.json
echo "=== retrieval_k3_chr_roberta ===";   run configs/roberta/retrieval/retrieval_k3_chr_roberta.json
echo "=== retrieval_k3_asc_roberta ===";   run configs/roberta/retrieval/retrieval_k3_asc_roberta.json
echo "=== retrieval_k4_chr_roberta ===";   run configs/roberta/retrieval/retrieval_k4_chr_roberta.json
echo "=== retrieval_k4_asc_roberta ===";   run configs/roberta/retrieval/retrieval_k4_asc_roberta.json
echo "=== retrieval_k5_chr_roberta ===";   run configs/roberta/retrieval/retrieval_k5_chr_roberta.json
echo "=== retrieval_k5_asc_roberta ===";   run configs/roberta/retrieval/retrieval_k5_asc_roberta.json

# C2 — retrieval (XLM-R)
echo "=== retrieval_k1_xlmr ===";         run configs/xlmr/retrieval/retrieval_k1_xlmr.json
echo "=== retrieval_k2_chr_xlmr ===";     run configs/xlmr/retrieval/retrieval_k2_chr_xlmr.json
echo "=== retrieval_k2_asc_xlmr ===";     run configs/xlmr/retrieval/retrieval_k2_asc_xlmr.json
echo "=== retrieval_k3_chr_xlmr ===";     run configs/xlmr/retrieval/retrieval_k3_chr_xlmr.json
echo "=== retrieval_k3_asc_xlmr ===";     run configs/xlmr/retrieval/retrieval_k3_asc_xlmr.json
echo "=== retrieval_k4_chr_xlmr ===";     run configs/xlmr/retrieval/retrieval_k4_chr_xlmr.json
echo "=== retrieval_k4_asc_xlmr ===";     run configs/xlmr/retrieval/retrieval_k4_asc_xlmr.json
echo "=== retrieval_k5_chr_xlmr ===";     run configs/xlmr/retrieval/retrieval_k5_chr_xlmr.json
echo "=== retrieval_k5_asc_xlmr ===";     run configs/xlmr/retrieval/retrieval_k5_asc_xlmr.json

# C3 — dual stream
echo "=== dual_stream_roberta ===";  run configs/roberta/dual_stream/dual_stream_roberta.json
echo "=== dual_stream_xlmr ===";     run configs/xlmr/dual_stream/dual_stream_xlmr.json

echo "=== ALL DONE ==="
