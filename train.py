"""
Alpha-MCP FunctionGemma Training Script
========================================
Standalone Python script — mirrors the notebook exactly.
Run this directly from the terminal without Jupyter.

USAGE:
    # Fresh training:
    python3 train.py

    # Resume from latest checkpoint:
    python3 train.py --resume

    # Custom epochs / output:
    python3 train.py --epochs 5 --output outputs/run2

    # Use Gemma 2 2B instead of 270M:
    python3 train.py --model google/gemma-2-2b-it
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULTS = {
    "model":       "google/functiongemma-270m-it",
    "dataset":     "data/synthetic_training.jsonl",
    "output":      "outputs/alpha_functiongemma",
    "checkpoints": "checkpoints",
    "n_samples":   1200,
    "epochs":      3,
    "lr":          2e-4,
    "batch":       1,
    "grad_accum":  8,
    "max_len":     512,
    "save_steps":  100,
    "warmup":      30,
    "lora_r":      16,
    "lora_alpha":  16,
    "lora_drop":   0.05,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train FunctionGemma on Alpha-MCP tools.")
    p.add_argument("--model",       default=DEFAULTS["model"])
    p.add_argument("--dataset",     default=DEFAULTS["dataset"])
    p.add_argument("--output",      default=DEFAULTS["output"])
    p.add_argument("--checkpoints", default=DEFAULTS["checkpoints"])
    p.add_argument("--epochs",      type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    p.add_argument("--max-len",     type=int,   default=DEFAULTS["max_len"])
    p.add_argument("--save-steps",  type=int,   default=DEFAULTS["save_steps"])
    p.add_argument("--resume",      action="store_true",
                   help="Auto-resume from latest checkpoint")
    p.add_argument("--regen-data",  action="store_true",
                   help="Re-run the data generator before training")
    p.add_argument("--n-samples",   type=int,   default=DEFAULTS["n_samples"],
                   help="Number of synthetic samples to generate (with --regen-data)")
    p.add_argument("--hf-token",    default=os.environ.get("HF_TOKEN", ""),
                   help="HuggingFace token (or set HF_TOKEN env var)")
    return p.parse_args()


def login_hf(token: str):
    if not token:
        print("⚠️  No HF_TOKEN provided. Attempting cached login...")
        try:
            from huggingface_hub import login
            login()
        except Exception:
            print("❌ HF login failed. If model is private/gated, set HF_TOKEN env var.")
            sys.exit(1)
    else:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ HF login OK")


def regen_data(n_samples: int, output_path: str):
    print(f"\n📊 Regenerating {n_samples} training samples...")
    script = os.path.join(os.path.dirname(__file__), "scripts", "generate_data.py")
    ret = os.system(f"python3 {script} --n {n_samples} --out {output_path}")
    if ret != 0:
        print("❌ Data generation failed.")
        sys.exit(1)


def find_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    cps = sorted(
        [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)
         if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1])
    )
    return cps[-1] if cps else None


TEST_PROMPTS = [
    "Give me the full analysis on NVDA.",
    "Run a diamond screen across the S&P 500.",
    "What's the current market regime?",
    "Risk audit my portfolio: AAPL, TSLA, MSFT.",
]


def run_eval(model, tokenizer, label: str):
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    model.eval()
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,  # Use greedy decoding for stability
                pad_token_id=tokenizer.eos_token_id
            )
        resp = tokenizer.decode(out[0], skip_special_tokens=False)
        has_call = "<start_function_call>" in resp
        status = "✅" if has_call else "❌"
        print(f"{status} {prompt}")
        if has_call:
            call = resp.split("<start_function_call>")[1].split("<end_function_call>")[0]
            print(f"   → {call}")
    model.train()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Alpha-MCP × FunctionGemma — Training Script       ║")
    print("╚══════════════════════════════════════════════════════╝\n")
    print(f"  Model      : {args.model}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Output     : {args.output}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Resume     : {args.resume}")
    print(f"  Regen data : {args.regen_data}\n")

    # ── GPU check ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires a GPU.")
        sys.exit(1)
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB VRAM)\n")

    # ── HuggingFace auth ─────────────────────────────────────────────────────
    login_hf(args.hf_token)

    # ── Optionally regenerate data ────────────────────────────────────────────
    if args.regen_data:
        regen_data(args.n_samples, args.dataset)

    if not os.path.exists(args.dataset):
        print(f"❌ Dataset not found: {args.dataset}")
        print("   Run: python3 scripts/generate_data.py  (or use --regen-data flag)")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"⏳ Loading {args.model}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Loaded in {time.time()-t0:.1f}s | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    # ── Pre-training eval ─────────────────────────────────────────────────────
    run_eval(model, tokenizer, "PRE-TRAINING BASELINE")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print("\n🛠️  Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=DEFAULTS["lora_r"],
        lora_alpha=DEFAULTS["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=DEFAULTS["lora_drop"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n📚 Loading dataset: {args.dataset}")
    raw = load_dataset("json", data_files=args.dataset, split="train")
    print(f"   {len(raw)} training examples")

    def fmt(example):
        text = ""
        for turn in example["conversations"]:
            text += f"<start_of_turn>{turn['role']}\n{turn['content']}<end_of_turn>\n"
        text += tokenizer.eos_token
        return {"text": text}

    dataset = raw.map(fmt, batched=False)

    # ── Checkpoint detection ─────────────────────────────────────────────────
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(args.checkpoints)
        if resume_checkpoint:
            print(f"\n▶️  Resuming from: {resume_checkpoint}")
        else:
            print("\n⚠️  No checkpoints found — starting fresh.")

    # ── Train ─────────────────────────────────────────────────────────────────
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_len,
            per_device_train_batch_size=DEFAULTS["batch"],
            gradient_accumulation_steps=DEFAULTS["grad_accum"],
            warmup_steps=DEFAULTS["warmup"],
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            bf16=False,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=args.checkpoints,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=5,
            report_to="none",
        ),
    )

    print(f"\n🚀 Training: {args.epochs} epoch(s) | {len(dataset)} samples | "
          f"checkpoint every {args.save_steps} steps\n")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed = time.time() - t0
    print(f"\n✅ Training complete in {elapsed/60:.1f} min")

    # ── Post-training eval ────────────────────────────────────────────────────
    run_eval(model, tokenizer, "POST-TRAINING TOOL CALL QUALITY")

    # ── Save final adapter ────────────────────────────────────────────────────
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    meta = {
        "base_model": args.model,
        "dataset": args.dataset,
        "num_samples": len(dataset),
        "epochs": args.epochs,
        "elapsed_min": round(elapsed / 60, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tools": ["analyze_ticker_full", "run_diamond_screen", "get_market_pulse", "run_risk_audit"],
    }
    with open(f"{args.output}/training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n🏁 Adapter saved → {args.output}")
    print(f"   Metadata   → {args.output}/training_metadata.json")
    print(f"\n🔍 To verify accuracy:")
    print(f"   python3 scripts/verify_model.py --model {args.output}")


if __name__ == "__main__":
    main()
