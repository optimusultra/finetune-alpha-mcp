"""
Verify Alpha-MCP Fine-tuned Model
===================================
Run after training to measure tool-calling accuracy.
Generates a score card with ✅/❌ for each test case.

USAGE:
    python scripts/verify_model.py --model outputs/alpha_functiongemma
"""

import torch
import json
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


TEST_CASES = [
    # (user_prompt, expected_tool_name)
    ("Give me the full technical breakdown on NVDA.", "analyze_ticker_full"),
    ("Is TSLA setting up for a breakout?", "analyze_ticker_full"),
    ("What stage is AMD in right now?", "analyze_ticker_full"),
    ("VCP check on AAPL.", "analyze_ticker_full"),
    ("Run a diamond screen across the S&P 500.", "run_diamond_screen"),
    ("Find the best setups in the market right now.", "run_diamond_screen"),
    ("Which of these qualify as diamonds? MSFT,GOOGL,META,AMZN", "run_diamond_screen"),
    ("Diamond screen the NASDAQ.", "run_diamond_screen"),
    ("What's the market regime?", "get_market_pulse"),
    ("Should I be risk-on or risk-off?", "get_market_pulse"),
    ("Is SPY above its 200 SMA?", "get_market_pulse"),
    ("Bull or bear market right now?", "get_market_pulse"),
    ("Run a risk audit on AAPL,TSLA,MSFT.", "run_risk_audit"),
    ("Check drawdown risk on my portfolio: NVDA, COIN, PLTR.", "run_risk_audit"),
    ("Are any of my positions down more than 10%? AMZN,GOOGL,META", "run_risk_audit"),
]


def run_verify(model_path: str, base_model: str = "google/functiongemma-270m-it"):
    print(f"Loading base: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    correct = 0
    results = []

    for prompt, expected in TEST_CASES:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0], skip_special_tokens=False)

        called_tool = None
        if "<start_function_call>" in resp:
            try:
                called_tool = resp.split("call:")[1].split("{")[0]
            except:
                pass

        ok = called_tool == expected
        if ok:
            correct += 1
        status = "✅" if ok else "❌"
        results.append({"prompt": prompt, "expected": expected, "got": called_tool, "ok": ok})
        print(f"{status} [{expected}] ← '{prompt}'")
        if not ok:
            print(f"   Got: {called_tool}")

    accuracy = correct / len(TEST_CASES)
    print(f"\n{'='*50}")
    print(f"Tool Call Accuracy: {correct}/{len(TEST_CASES)} = {accuracy:.1%}")
    if accuracy >= 0.85:
        print("🏆 PASS — Model is ready for deployment!")
    elif accuracy >= 0.60:
        print("⚠️  PARTIAL — Consider more training epochs or samples.")
    else:
        print("❌ FAIL — Model needs more training.")

    # Save results
    with open("verification_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "cases": results}, f, indent=2)
    print(f"\nResults saved: verification_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned adapter")
    parser.add_argument("--base", default="google/functiongemma-270m-it", help="Base model ID")
    args = parser.parse_args()
    run_verify(args.model, args.base)
