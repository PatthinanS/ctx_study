# Context Selection & VAD Trajectory Study

Investigates how conversational context selection strategy affects VAD (Valence–Arousal–Dominance) emotional trajectories in dialogue. Runs RoBERTa and XLM-R VAD regression under four context conditions and compares per-turn VAD trajectories across conditions.

---

## Setup

```bash
bash setup.sh
```

For GPU support, install the matching torch variant afterwards:
```bash
# CUDA 11.8
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Place the IEMOCAP CSV at:
```
data/iemocap/iemocap_merged_all.csv
```

Required columns: `session, dialog, utterance_id, speaker, start_time, text, valence, arousal, dominance`

---

## Running Experiments

### Single experiment
```bash
python main.py --config configs/roberta/base/base_roberta.json
```

### All experiments (sequential)
```bash
bash run_experiments.sh
```

### Quick sanity check (1 epoch, 64 samples, CPU)
```bash
python main.py --config configs/roberta/base/base_roberta.json --smoke
```

### Compare C1 vs C2 context side-by-side
```bash
python main.py --config configs/roberta/retrieval/retrieval_k3_chr_roberta.json --sanity
```

### Override config fields at runtime
```bash
python main.py --config configs/roberta/base/base_roberta.json --override phase2.epochs=5 batch_size=32
```

---

## Experimental Conditions

| Condition | Strategy | Configs |
|-----------|----------|---------|
| C0 — no context | Target utterance only | `roberta/base/`, `xlmr/base/` |
| C1 — window | Last k=3 turns prepended | `roberta/window/`, `xlmr/window/` |
| C2 — retrieval | Top-k similar turns by cosine sim | `roberta/retrieval/`, `xlmr/retrieval/` |
| C3 — dual stream | Same-speaker + cross-speaker streams fused | `roberta/dual_stream/`, `xlmr/dual_stream/` |

All conditions share the same backbone, training loop, and MSE loss. Only the context varies.

---

## Config Structure

```
configs/
  roberta/
    base/          base_roberta.json               ← C0
    window/        context_roberta.json             ← C1
    retrieval/     retrieval_k{1,2,3,4,5}_{chr,asc}_roberta.json  ← C2
    dual_stream/   dual_stream_roberta.json         ← C3
  xlmr/
    base/          base_xlmr.json
    window/        context_xlmr.json
    retrieval/     retrieval_k{1,2,3,4,5}_{chr,asc}_xlmr.json
    dual_stream/   dual_stream_xlmr.json
```

### Key config fields

```json
{
  "backbone": "roberta-base",
  "context": {
    "strategy": "none | window | retrieval | dual_stream",
    "k": 3,
    "strategy_kwargs": { "sort_by": "chrono | sim_asc" }
  },
  "phase1": { "epochs": 5,  "lr": 3e-3, "freeze_encoder": true  },
  "phase2": { "epochs": 15, "lr": 5e-6, "freeze_encoder": false, "head_lr_multiplier": 10 },
  "splits": { "mode": "leave_one_session_out", "test_session": 5, "val_session": 4 }
}
```

**Retrieval sort orders:**
- `chrono` — top-k turns sorted by original dialogue order
- `sim_asc` — top-k turns sorted by ascending similarity (least similar first, closest to target last) (`asc` still accepted as a deprecated alias)

---

## Training

Two-phase training per experiment:

| Phase | Epochs | Encoder | Learning rate |
|-------|--------|---------|---------------|
| 1 | 5 | Frozen | 3e-3 (head only) |
| 2 | 15 | Unfrozen | 5e-6 (backbone), 5e-5 (head, 10× multiplier) |

Best checkpoint selected by validation mean Pearson r. Loaded automatically before test evaluation.

---

## Outputs

Each experiment writes to `outputs/<experiment_name>/`:

```
outputs/base_roberta/
  best.pt              ← best val checkpoint
  last.pt              ← final epoch checkpoint
  training_log.csv     ← per-epoch metrics
```

`training_log.csv` columns: `phase, epoch, train_loss, r_valence, r_arousal, r_dominance, mean_pearson, best_so_far`

---

## C3 Architecture (dual stream)

Unlike C0–C2 which prepend context as text, C3 uses three separate backbone passes:

```
target utterance        → backbone → target_repr  (H)
same-speaker history    → backbone → same_repr    (H)   ← zeroed if no prior turns
cross-speaker history   → backbone → cross_repr   (H)   ← zeroed if no prior turns

concat([target, same, cross])  →  Linear(3H → H)  →  VADHead  →  [V, A, D]
```

Each stream is all prior turns from that speaker concatenated into one sequence and encoded once. The fusion layer (`Linear(3H, H)`) is the only architectural addition; backbone and VAD head are unchanged from C0–C2.

---

## Evaluation

Primary metric: **Pearson r** per VAD dimension (valence, arousal, dominance) + mean.

Post-training trajectory analysis (planned): UED metrics + pairwise DTW on per-turn VAD sequences.

---

## Project Structure

```
main.py              entry point
run_experiments.sh   runs all configs sequentially
src/
  data.py            IEMOCAP loader, context strategies, Dataset, DataLoaders
  model.py           VADModel, VADModelDualStream, VADHead
  train.py           two-phase training loop, checkpointing, evaluation
  metrics.py         Pearson r per VAD dimension
configs/             JSON configs (see above)
data/iemocap/        CSV data (not tracked)
outputs/             per-experiment results (not tracked)
```
