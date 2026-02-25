# 🦞 `finetune_alpha_mcp` — Alpha-MCP × FunctionGemma Fine-tuning

> Fine-tune **Google FunctionGemma 270M** (or Gemma 2 2B) to natively trigger `alpha-mcp` tools — turning a plain language request like *"How is NVDA looking?"* into a perfectly structured tool call with zero hallucination.

---

## 🧠 What This Does

The `alpha-mcp` server exposes 4 precision tools for institutional-grade market analysis. Out-of-the-box, general language models don't know these tools exist. This package fine-tunes a tiny, fast model to **reliably and unambiguously dispatch the correct tool** given a natural language prompt — with proper arguments, correct schema, and no confusion between similar intents.

**Target accuracy: ≥ 85% on the 15-prompt eval suite.**

---

## 📦 Package Structure

```
finetune_alpha_mcp/
├── train.py                           ← Standalone CLI training script
├── DEPLOYMENT.md                      ← Basic deployment options
├── INTEGRATION_PLAN.md                ← Detailed Skill + Micro-script architecture
├── tool_schemas.py                    ← Single source of truth for ALL tool definitions
├── pyproject.toml                     ← uv project config (dependency management)
├── setup_env.sh                       ← One-shot bootstrap for the isolated environment
├── requirements.txt                   ← Alternative pip requirements
│
├── train_alpha_functiongemma.ipynb    ← Interactive training notebook (same as train.py)
│
├── scripts/
│   ├── generate_data.py               ← Synthetic training data generator (1200 examples)
│   └── verify_model.py                ← Post-training accuracy scorecard
│
├── data/
│   ├── synthetic_training.jsonl       ← Pre-generated training set (auto-created)
│   └── tool_api_contract.json         ← Human-readable summary of all tool signatures
│
├── checkpoints/                       ← Auto-saved training checkpoints (gitignored)
└── outputs/
    └── alpha_functiongemma/           ← Final LoRA adapter (gitignored)
```

---

## 🚀 Quickstart

### 1. Set Up the Isolated Environment (uv)

```bash
cd /home/mihir/projects/finetune_alpha_mcp

# Create .venv + install CUDA-enabled PyTorch + ML stack:
bash setup_env.sh
```

> This creates a `.venv` directory isolated from your system Python and installs all dependencies (torch, transformers, trl, peft, datasets, accelerate) pinned to compatible versions.

### 2. Accept the HuggingFace License

`FunctionGemma 270M` is a **gated model**. You must:
1. Go to [huggingface.co/google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
2. Click **"Access Repository"** and accept the license.
3. Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
export HF_TOKEN=hf_your_token_here
```

### 3. Generate Synthetic Training Data

```bash
uv run python3 scripts/generate_data.py --n 1200
```

> Or just run `train.py` with the `--regen-data` flag to do it automatically.

### 4. Train

```bash
# Default: 3 epochs, FunctionGemma 270M, checkpoint every 100 steps
uv run python3 train.py

# Resume from last checkpoint (after a crash or interruption):
uv run python3 train.py --resume

# More epochs for better accuracy:
uv run python3 train.py --epochs 5

# Re-generate fresh data + train in one shot:
uv run python3 train.py --regen-data --epochs 3
```

### 5. Verify

```bash
uv run python3 scripts/verify_model.py --model outputs/alpha_functiongemma
```

Expected output:
```
✅ [analyze_ticker_full] ← 'Give me the full analysis on NVDA.'
✅ [run_diamond_screen]  ← 'Run a diamond screen across the S&P 500.'
✅ [get_market_pulse]    ← 'What's the current market regime?'
✅ [run_risk_audit]      ← 'Risk audit my portfolio: AAPL, TSLA, MSFT.'
...
Tool Call Accuracy: 14/15 = 93.3%
🏆 PASS — Model is ready for deployment!
```

---

## 🛠 Training Configuration

Key parameters in `train.py` (all overridable via CLI):

| Parameter | Default | Description |
|:---|:---|:---|
| `--model` | `google/functiongemma-270m-it` | Base model |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--max-len` | `512` | Max token length per sample |
| `--save-steps` | `100` | Checkpoint frequency |
| `--resume` | `False` | Auto-resume from latest checkpoint |
| `--regen-data` | `False` | Regenerate synthetic data before training |

### LoRA Configuration (in `DEFAULTS`)

```python
lora_r       = 16   # Rank — increase to 32 for more capacity
lora_alpha   = 16   # Scaling
lora_dropout = 0.05 # Regularization
```

### VRAM Budget (GTX 1060 6GB)

| Stage | VRAM |
|:---|:---|
| Base model loaded | ~1.0 GB (270M fp16) |
| + LoRA adapters | ~1.2 GB |
| + Training batch | ~2.5–3.5 GB |
| Peak | ≤ 5 GB ✅ |

---

## 📊 Synthetic Dataset

The data generator (`scripts/generate_data.py`) produces **1,200 realistic training conversations** weighted by expected real-world usage:

| Tool | Samples | % |
|:---|:---:|:---:|
| `analyze_ticker_full` | ~492 | 41% |
| `run_diamond_screen` | ~338 | 28% |
| `get_market_pulse` | ~251 | 21% |
| `run_risk_audit` | ~119 | 10% |
| Disambiguation (multi-intent) | ~60 | 5% |

**Tickers used**: 40 real symbols across S&P 500 mega-caps, NASDAQ 100, mid-cap, and Russell small-caps — ensuring the model generalizes beyond just `AAPL` and `TSLA`.

**Format**: ShareGPT → FunctionGemma `<start_of_turn>` / `<start_function_call>call:name{...}<end_function_call>` token sequence.

---

## ♻️ Expanding When Alpha-MCP Grows

When `server.py` adds a new tool, the model needs to learn it. Here's the workflow to continue training without losing what the model already knows:

### Step 1 — Register the new tool in `tool_schemas.py`
```python
# In the EXPANSION ZONE at the bottom of ALPHA_MCP_TOOLS:
{
    "name": "get_earnings_calendar",
    "description": "Returns upcoming earnings dates and consensus estimates for a ticker.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock symbol"},
            "weeks_ahead": {"type": "integer", "description": "How many weeks to look ahead"}
        },
        "required": ["ticker"]
    }
}
```

### Step 2 — Add prompt templates in `scripts/generate_data.py`
```python
def gen_get_earnings_calendar():
    t = random.choice(ALL_TICKERS)
    templates = [
        f"When does {t} report earnings?",
        f"Check the earnings calendar for {t}.",
        ...
    ]
    return build_sample(random.choice(templates), "get_earnings_calendar", {"ticker": t})
```

### Step 3 — Resume train on expanded dataset
```bash
uv run python3 train.py --regen-data --resume --epochs 2
```

> `--resume` ensures the model continues from the last checkpoint, not from scratch. The existing tool knowledge is preserved.

---

## 🔍 Model Comparison

| Base Model | Size | VRAM | Speed | Tool Accuracy (est.) |
|:---|:---:|:---:|:---:|:---:|
| `functiongemma-270m-it` (**default**) | 270M | ~2.5 GB | Fast | ~85–92% |
| `gemma-2-2b-it` | 2B | ~4.5 GB | Slower | ~90–96% |

Switch to 2B for higher accuracy at the cost of slower inference:
```bash
uv run python3 train.py --model google/gemma-2-2b-it --epochs 3
```

---

## 🔗 Related

- **Alpha-MCP Server**: `/home/mihir/projects/openclaw/mcp-servers/alpha-mcp/`
- **FunctionGemma HF Page**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
- **FunctionGemma Docs**: [ai.google.dev](https://ai.google.dev/gemma/docs/function_gemma)

---

*Built for the Optimus Intelligence Workstation. 🦞🛡️📈*
