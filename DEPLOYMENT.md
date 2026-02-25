# 🛳️ Deployment & Integration Guide — Alpha-MCP × OpenClaw

This guide explains how to take your finished LoRA adapter from `outputs/alpha_functiongemma/` and integrate it into the OpenClaw agent fleet.

---

## 🏗 Option 1: Local Python Integration (Fastest)

If you want to use the model directly in an OpenClaw agent (e.g., as a local decision-engine for the `Musings` strategist), use the `Transformers` + `PEFT` pattern.

### 1. Load the Adapter in Python
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "google/functiongemma-270m-it"
ADAPTER_PATH = "/home/mihir/projects/finetune_alpha_mcp/outputs/alpha_functiongemma"

# 1. Load Base
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. Add Alpha-MCP LoRA Adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# 3. Usage inside an Agent
def get_alpha_tool_call(query):
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

---

## 🦙 Option 2: Ollama Integration (Recommended for Microservices)

To expose the fine-tuned model as a local API that multiple agents can call, use Ollama.

### 1. Export as GGUF (Optional but Recommended)
For high-speed inference in Ollama, you usually want to merge the LoRA weights and quantize to GGUF. You can use tools like `llama.cpp` to convert.

### 2. Create a Modelfile
If you are running the un-merged model via a local Python server like TGI or vLLM, you can simply point your agents to that endpoint. 

If using Ollama with a merged model:
```dockerfile
FROM ./merged_alpha_gemma.gguf
PARAMETER temperature 0.1
PARAMETER top_p 0.95
SYSTEM "You are the Alpha-MCP Dispatcher. Your only job is to output tool calls in the <start_function_call> format."
```

---

## 🤖 OpenClaw Agent Integration

To make an agent like `Musings` use this model, update your `openclaw.json` or the agent's `SOUL.md`.

### 1. Update `openclaw.json`
Configure the agent to hit your local endpoint for tool-calling decisions.

```json
{
  "id": "musings-alpha-brain",
  "model": {
    "primary": "local/alpha-functiongemma",
    "fallbacks": ["openrouter/free"]
  },
  "endpoint": "http://localhost:11434/v1" 
}
```

### 2. Prompt Engineering for Decision Making
In the `Musings` agent's `IDENTITY.md`, instruct it to utilize the Alpha-MCP tools whenever relevant market activity is detected:

```markdown
Whenever the user mentions a stock ticker or "market sentiment", 
use the analyze_ticker_full or get_market_pulse tool to 
inform your strategist notes.
```

---

## 📊 Performance Tuning

### VRAM Optimization
The **270M** model is tiny. If you run it alongside other models:
- It only uses **~2.5GB VRAM** during inference.
- You can keep it "Always-On" via `ollama run --keep-alive -1 alpha-mcp`.

### Latency
- FunctionGemma 270M inference is nearly instantaneous (< 100ms on GTX 1060).
- Perfect for real-time monitoring where the model needs to decide if a trade alert should be fired.

---

### Workflow Checklist for New Tools
Whenever you expand `alpha-mcp` with a new tool:
1. 🛠 **Update** `tool_schemas.py`
2. 🧪 **Generate** new data (`scripts/generate_data.py`)
3. 🏃 **Retrain** (`python3 train.py --resume`)
4. 🛳 **Redeploy** the new adapter to your agent's model directory.

*Integration complete. 🦞🤖🛡️*
