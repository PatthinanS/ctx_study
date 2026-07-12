# Research: Conversational Context Selection & VAD Emotional Trajectories

## Objective

Investigate how conversational context selection strategy affects VAD (Valence-Arousal-Dominance) emotional trajectories in dialogue. Run the XLM-R and RoBERTa VAD regression pipeline under four context conditions and measure how the resulting per-turn VAD trajectories differ. Loss: MSE only.

## Four Experimental Conditions

| ID | Name | Status | Description |
|----|------|--------|-------------|
| C0 | None | Implemented | Target utterance only — baseline |
| C1 | Window | Implemented | Last k=3 turns prepended chronologically |
| C2 | Retrieval | Implemented | Top-k turns by cosine sim, sorted `chrono` or `asc` |
| C3 | Speaker-stratified dual-stream | **NOT YET IMPLEMENTED** | Same-speaker + cross-speaker streams encoded separately, fused via concat+linear before VAD head |

## Project Structure

```
main.py                  # Entry point: load config → train(cfg, device)
run_experiments.sh       # Shell script to run all configs sequentially
configs/                 # One JSON per experiment variant (see below)
src/
  data.py                # IEMOCAP loader, context strategies, Dataset, DataLoaders
  model.py               # VADModel (backbone + VADHead), loss functions
  train.py               # Two-phase training loop, eval, checkpointing
  metrics.py             # Pearson r per V/A/D dimension
outputs/                 # Per-experiment: best.pt, last.pt, training_log.csv
data/iemocap/            # iemocap_merged_all.csv (required columns below)
```

## Config Files (maps to conditions)

| Config | Condition | Backbone |
|--------|-----------|----------|
| `base_roberta.json` | C0 | RoBERTa |
| `base_xlmr.json` | C0 | XLM-R |
| `context_roberta.json` | C1 (window k=3) | RoBERTa |
| `context_xlmr.json` | C1 (window k=3) | XLM-R |
| `retrieval_k1_roberta.json` | C2 (k=1) | RoBERTa |
| `retrieval_k3_chr_roberta.json` | C2 (k=3, chrono) | RoBERTa |
| `retrieval_k3_asc_roberta.json` | C2 (k=3, asc) | RoBERTa |
| *(xlmr variants mirror above)* | | XLM-R |

## Model Architecture (`src/model.py`)

```
Input text → Backbone (RoBERTa / XLM-R)
           → CLS / pooler_output (hidden_size)
           → Dropout → Linear(hidden_size, 3)
           → [valence, arousal, dominance]  ← MSE vs ground truth
```

- `VADModel`: backbone + `VADHead`
- `freeze_encoder()` / `unfreeze_encoder()` used between phases
- `EMDLoss` and `CombinedLoss` are carried over from a prior VAD pipeline and remain available; this study fixes loss to MSE so context condition is the only variable

## Training Loop (`src/train.py`)

Two-phase training per experiment:
- **Phase 1** (5 epochs): encoder frozen, head-only training, lr=3e-3
- **Phase 2** (15 epochs): full fine-tune, backbone lr=5e-6, head lr=5e-5 (10× multiplier)
- Best checkpoint saved by val `mean_pearson`; loaded before test eval
- Outputs: `best.pt`, `last.pt`, `training_log.csv` per `output_dir`

## Data (`src/data.py`)

- Dataset: IEMOCAP — CSV with columns: `session, dialog, utterance_id, speaker, start_time, text, valence, arousal, dominance`
- Split: leave-one-session-out (sessions 1–3 train, 4 val, 5 test)
- Context strategies registered in `STRATEGY_REGISTRY`:
  - `"none"` → returns `(None, cur_str)`
  - `"window"` → last k turns as `ctx_str`
  - `"retrieval"` → top-k by cosine sim on backbone embeddings; `sort_by="chrono"` or `"sim_asc"` (`"asc"` deprecated alias)
- For retrieval: dialogue embeddings computed once per dialogue via mean-pool of last hidden state
- Tokenization: `tokenizer(ctx, cur)` for context conditions; `tokenizer(cur)` for C0

## Evaluation (`src/metrics.py`)

- Primary: Pearson r per V/A/D + `mean_pearson`
- Post-training (planned): UED metrics + pairwise DTW on VAD trajectories

## CLI

```bash
python main.py --config configs/base_roberta.json
python main.py --config configs/context_roberta.json --smoke   # fast CPU test
python main.py --config configs/context_roberta.json --sanity  # print C1 vs C2 context diff
python main.py --config configs/base_roberta.json --override phase2.epochs=5
```

## What's Missing / Next

- **C3 implementation**: needs architecture change in `src/model.py` (dual-stream encoder + fusion layer) and a new context strategy or dataset variant in `src/data.py`
- Post-training trajectory analysis (UED + DTW) not yet implemented
