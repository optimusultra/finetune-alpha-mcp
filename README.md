# Alpha-MCP Fine-tuning Package

Trains **FunctionGemma 270M** (or Gemma 2 2B) to natively call `alpha-mcp` tools with zero ambiguity.

## Structure

```
finetune_alpha_mcp/
├── tool_schemas.py                    # Single source of truth for all tools
├── train_alpha_functiongemma.ipynb    # Checkpoint-aware training notebook
├── scripts/
│   ├── generate_data.py               # Synthetic data generator (1200+ examples)
│   └── verify_model.py                # Post-training accuracy scorecard
├── data/
│   ├── tool_api_contract.json         # Human-readable API contract
│   └── synthetic_training.jsonl       # Generated — run generate_data.py
├── checkpoints/                       # Auto-saved training checkpoints
└── outputs/
    └── alpha_functiongemma/           # Final saved adapter
```

## Quickstart

### 1. Generate Training Data
```bash
python scripts/generate_data.py --n 1200
```

### 2. Train
Open `train_alpha_functiongemma.ipynb` and run all cells.  
Set `RESUME_FROM_CHECKPOINT = True` to continue from a saved checkpoint.

### 3. Verify
```bash
python scripts/verify_model.py --model outputs/alpha_functiongemma
```
Target: ≥ 85% tool-call accuracy.

## Expanding When Alpha-MCP Grows

When `server.py` adds a new tool:
1. Add the JSON schema to `tool_schemas.py` (in the EXPANSION ZONE)
2. Add example templates to `scripts/generate_data.py`
3. Set `RESUME_FROM_CHECKPOINT = True` in the notebook
4. Re-run generate + train — the model continues learning without starting over

## Hardware Requirements
- GTX 1060 6GB (fp16, batch=1, max_length=512)
- Python 3.10 + `unsloth_env` kernel
- HuggingFace account with FunctionGemma license accepted
