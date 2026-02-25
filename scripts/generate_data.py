"""
Synthetic Data Generator for Alpha-MCP Fine-tuning
====================================================
Generates 1000+ diverse training examples in FunctionGemma's tool-calling format.
Each example is a realistic conversation that an OPTIMUS agent would have,
leading to a precise alpha-mcp tool call.

HOW TO RUN:
    python scripts/generate_data.py

HOW TO EXPAND:
    When alpha-mcp adds new tools, add their schemas to tool_schemas.py
    and add new example templates to the EXAMPLES dict. Then re-run this script.
    The new dataset combines with checkpoints so you can keep retraining.
"""

import json
import random
import os
import sys

# Add parent path so we can import tool_schemas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool_schemas import ALPHA_MCP_TOOLS

# ─── Real stock universes for realistic prompts ───────────────────────────────
SP500_MEGA = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "BRK.B", "JPM", "V"]
SP500_MID  = ["SQ", "COIN", "PLTR", "RBLX", "SNAP", "UBER", "LYFT", "ZM", "DKNG", "PTON"]
NASDAQ_100 = ["AMD", "QCOM", "INTC", "MU", "MRVL", "AVGO", "AMAT", "KLAC", "LRCX", "CDNS"]
RUSSELL_SM = ["ACMR", "APLS", "CRDO", "LADD", "TCMD", "HIMS", "SKIN", "FWRG", "MYMD", "MIST"]
ALL_TICKERS = SP500_MEGA + SP500_MID + NASDAQ_100 + RUSSELL_SM

INDICES   = ["sp500", "nasdaq", "russell2000", "both"]
SECTORS   = ["semiconductors", "biotech", "fintech", "EV", "cloud", "defense", "energy", "AI"]
POSITIONS = ["position", "swing trade", "holding", "portfolio position", "trade"]

random.seed(42)


def fmt_tool_call(name: str, args: dict) -> str:
    """Formats a tool call in the FunctionGemma <start_function_call> format."""
    args_str = json.dumps(args, separators=(", ", ": "))
    return f"<start_function_call>call:{name}{args_str}<end_function_call>"


def build_sample(user_msg: str, tool_name: str, tool_args: dict) -> dict:
    """Builds a single training sample in ShareGPT format compatible with SFTTrainer."""
    return {
        "conversations": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": fmt_tool_call(tool_name, tool_args)},
        ]
    }


# ─── Template pools for each tool ─────────────────────────────────────────────

def gen_analyze_ticker():
    """Generate examples for analyze_ticker_full."""
    t = random.choice(ALL_TICKERS)
    templates = [
        f"What's the technical picture on {t}?",
        f"Give me the full breakdown on {t}.",
        f"Can you vet {t} for me? Full analysis.",
        f"How is {t} looking right now?",
        f"Run a deep dive on {t}.",
        f"I'm thinking about adding {t}. What does the chart say?",
        f"Does {t} look like a buy setup?",
        f"Give me the Diamond score and stage for {t}.",
        f"I heard {t} is breaking out. Confirm?",
        f"Is {t} in a Stage 2 uptrend?",
        f"Show me the VCP pattern status on {t}.",
        f"What's the RSI and volume profile for {t}?",
        f"Relative strength check on {t} vs SPY.",
        f"Is there institutional flow into {t}?",
        f"Scan {t} for breakout confirmation.",
        f"What are the support and resistance levels on {t}?",
        f"Run the Alpha-MCP screen on {t}.",
        f"Vet {t} against the Weinstein methodology.",
        f"Full RSNHBP analysis on {t}, please.",
        f"I want to enter {t}. What does alpha-mcp say?",
        f"How many signals is {t} triggering?",
        f"Any short squeeze potential in {t}?",
        f"Check the PEG ratio and value signals on {t}.",
        f"Is the mean reversion setup intact for {t}?",
        f"Walk me through every strategy flag on {t}.",
    ]
    return build_sample(random.choice(templates), "analyze_ticker_full", {"ticker": t})


def gen_run_diamond_screen():
    """Generate examples for run_diamond_screen."""
    samples = []

    # 1. Whole-market scan (no tickers)
    idx = random.choice(INDICES)
    whole_market_templates = [
        f"Run a diamond screen across {idx}.",
        f"What are the top Diamond setups in the market right now ({idx})?",
        f"Find me the best breakouts in {idx} today.",
        f"Show me the whole-market Diamond scan.",
        f"Any Diamond-quality setups in {idx}?",
        f"Give me the Diamond leaderboard from the {idx} scan.",
        f"Set the index to {idx} and do a full diamond screen.",
        f"Which stocks are hitting Diamond status in {idx} right now?",
        f"Pull the top 20 diamond setups from {idx}.",
        f"I want a market-wide scan of {idx} for diamond signals.",
        f"Find me the alpha in {idx} — full diamond screen.",
        f"Run the cache scan on {idx} for diamonds.",
    ]
    samples.append(build_sample(random.choice(whole_market_templates), "run_diamond_screen", {"index": idx}))

    # 2. Specific ticker batch screen
    n = random.randint(2, 6)
    tickers = random.sample(ALL_TICKERS, n)
    ticker_str = ",".join(tickers)
    batch_templates = [
        f"Diamond screen this batch: {ticker_str}.",
        f"Which of these are diamonds? {ticker_str}",
        f"Run diamond filter on {ticker_str}.",
        f"Check if any of these qualify as Diamond: {ticker_str}.",
        f"Screen {ticker_str} for Diamond-quality setups.",
        f"Are there any Diamond setups among {ticker_str}?",
        f"Give me the alpha ranking for {ticker_str}.",
    ]
    samples.append(build_sample(random.choice(batch_templates), "run_diamond_screen", {"tickers": ticker_str, "index": idx}))

    return random.choice(samples)


def gen_get_market_pulse():
    """Generate examples for get_market_pulse."""
    templates = [
        "What's the current market regime?",
        "Should I be in offense or defense mode right now?",
        "How is the macro looking? Bull or bear?",
        "Get me the market pulse.",
        "Is SPY above or below its key moving averages?",
        "What's the overall market trend score?",
        "Are we in a BULLISH or BEARISH regime right now?",
        "What should my market posture be today?",
        "Give me the macro health check.",
        "Is the market in a buy or sell environment?",
        "Bull market or not? Check the pulse.",
        "Should I be risk-on or risk-off?",
        "What's the broad market doing vs its SMAs?",
        "Give me the market breadth regime check.",
        "How is the tape looking overall?",
        "Is this a 'go big' or 'stay cautious' market?",
        "Check SPY vs 20/50/200 SMA for me.",
        "Are we on offense or defense based on the trend score?",
        "Before I scan for winners, what's the market backdrop?",
        "Macro check — before I put on trades today.",
        "Run a quick pulse check on the broader market.",
        "What's the regime? I need to calibrate my risk.",
        "BULLISH, CAUTION, or BEARISH — where are we?",
    ]
    return build_sample(random.choice(templates), "get_market_pulse", {})


def gen_run_risk_audit():
    """Generate examples for run_risk_audit."""
    n = random.randint(2, 7)
    tickers = random.sample(ALL_TICKERS, n)
    ticker_str = ",".join(tickers)
    pos_word = random.choice(POSITIONS)
    templates = [
        f"Run a risk audit on my {pos_word}: {ticker_str}.",
        f"Check drawdown risk on {ticker_str}.",
        f"Are any of these flashing risk alerts? {ticker_str}",
        f"Audit {ticker_str} — I need to know if anything is in danger.",
        f"How much has {ticker_str} pulled back from highs?",
        f"Is my {pos_word} in {ticker_str} down more than 10%?",
        f"Risk check on {ticker_str} before market open.",
        f"Do I need to stop out of anything in {ticker_str}?",
        f"Give me the drawdown table for {ticker_str}.",
        f"Full risk audit: {ticker_str}.",
        f"Any of these crossed the 10% drawdown alert threshold? {ticker_str}",
        f"Portfolio risk check — {ticker_str}.",
        f"Protect the book — audit {ticker_str} for risk.",
    ]
    return build_sample(random.choice(templates), "run_risk_audit", {"tickers": ticker_str})


# ─── Meta-level disambiguation examples ───────────────────────────────────────

def gen_disambiguation():
    """
    Tricky prompts that could map to multiple tools.
    Forces the model to develop intent-awareness.
    """
    disambigs = [
        # "how is X?" → analyze_ticker not market pulse
        (f"How is {random.choice(ALL_TICKERS)} looking this week?",
         "analyze_ticker_full", {"ticker": random.choice(ALL_TICKERS)}),

        # "market" + "specific ticker" → pulse first, then analyze
        ("Before I check NVDA, what's the broad market regime?",
         "get_market_pulse", {}),

        # "screen" with many tickers → run_diamond_screen
        (f"Screen {','.join(random.sample(ALL_TICKERS, 4))} — who qualifies?",
         "run_diamond_screen", {"tickers": ",".join(random.sample(ALL_TICKERS, 4))}),

        # risk focus → run_risk_audit
        (f"I'm worried about my {random.choice(POSITIONS)} in {','.join(random.sample(ALL_TICKERS, 3))}. Check the risk.",
         "run_risk_audit", {"tickers": ",".join(random.sample(ALL_TICKERS, 3))}),

        # Whole market diamond — no ticker mentioned
        ("Find me the best Diamond setups right now, market-wide.",
         "run_diamond_screen", {"index": "both"}),

        # Stage 2 + VCP → analyze
        (f"Is {random.choice(ALL_TICKERS)} in Stage 2 with a VCP setup?",
         "analyze_ticker_full", {"ticker": random.choice(ALL_TICKERS)}),

        # "is now a good time" → pulse
        ("Is now a good time to buy breakouts?",
         "get_market_pulse", {}),
    ]
    item = random.choice(disambigs)
    return build_sample(item[0], item[1], item[2])


# ─── Main generator ───────────────────────────────────────────────────────────

GENERATORS = [
    (gen_analyze_ticker, 0.40),        # 40% → most common use case
    (gen_run_diamond_screen, 0.25),    # 25% → screen mode
    (gen_get_market_pulse, 0.20),      # 20% → regime check
    (gen_run_risk_audit, 0.10),        # 10% → risk audits
    (gen_disambiguation, 0.05),        # 5%  → hard disambiguation
]


def pick_generator():
    r = random.random()
    cumulative = 0.0
    for gen_fn, weight in GENERATORS:
        cumulative += weight
        if r <= cumulative:
            return gen_fn
    return GENERATORS[-1][0]


def generate_dataset(n_samples: int = 1200, output_path: str = "data/synthetic_training.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    samples = []
    for i in range(n_samples):
        gen_fn = pick_generator()
        samples.append(gen_fn())

    random.shuffle(samples)

    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Generated {n_samples} samples → {output_path}")
    print(f"   Breakdown:")
    counter = {}
    for s in samples:
        name = s["conversations"][1]["content"].split("call:")[1].split("{")[0]
        counter[name] = counter.get(name, 0) + 1
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"   {k}: {v} ({v/n_samples:.1%})")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Alpha-MCP training data.")
    parser.add_argument("--n", type=int, default=1200, help="Number of samples to generate (default: 1200)")
    parser.add_argument("--out", type=str, default="data/synthetic_training.jsonl", help="Output file")
    args = parser.parse_args()
    generate_dataset(n_samples=args.n, output_path=args.out)
