#!/bin/bash
# C0 / C1 / C2(k=4, chrono) / C3 batch — RoBERTa only.
# C3's stream_k is overridden to 4 so its per-stream context size matches C2's k=4.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON=.venv/bin/python
LOG="run_log_c0_c1_c2k4_c3_$(date +%Y%m%d_%H%M%S).txt"

run() {
  local name="$1" cfg="$2"
  shift 2
  echo "=== $name ===" | tee -a "$LOG"
  $PYTHON main.py --config "$cfg" "$@" 2>&1 | tee -a "$LOG"
}

# C0 — no context
run base_roberta configs/roberta/base/base_roberta.json

# C1 — window (k=3)
run context_roberta configs/roberta/window/context_roberta.json

# C2 — retrieval, k=4, chrono
run retrieval_k4_chr_roberta configs/roberta/retrieval/retrieval_k4_chr_roberta.json

# C3 — dual stream, stream_k overridden to 4 to match C2's context size
run dual_stream_roberta configs/roberta/dual_stream/dual_stream_roberta.json --override context.stream_k=4

echo "=== ALL DONE ===" | tee -a "$LOG"
