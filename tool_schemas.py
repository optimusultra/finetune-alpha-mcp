"""
Alpha-MCP Tool Schemas — Single Source of Truth
================================================
All tool definitions live here. When alpha-mcp expands with new tools,
add them to ALPHA_MCP_TOOLS and the synthetic data generator will automatically
pick them up on the next training run.
"""

ALPHA_MCP_TOOLS = [
    {
        "name": "analyze_ticker_full",
        "description": (
            "Performs a comprehensive technical analysis of a single stock ticker. "
            "Returns Weinstein Stage, Diamond status, VCP, breakout, relative strength, "
            "institutional flow, RSI, SMAs, support/resistance, and volume stats."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock symbol to analyze, e.g. 'AAPL', 'NVDA', 'TSLA'."
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "run_diamond_screen",
        "description": (
            "Scans the market for Diamond-status stocks — setups with the highest alpha potential. "
            "Can scan an explicit list of tickers OR sweep the entire hot-loaded cache for whole-market "
            "Diamond discoveries. Returns a ranked list of up to 20 top setups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "string",
                    "description": "Comma-separated tickers to screen, e.g. 'AAPL,NVDA,MSFT'. Leave empty for whole-market cache scan."
                },
                "index": {
                    "type": "string",
                    "description": "Index to scan: 'sp500', 'nasdaq', 'russell2000', or 'both'. Defaults to 'both'.",
                    "enum": ["sp500", "nasdaq", "russell2000", "both"]
                }
            },
            "required": []
        }
    },
    {
        "name": "get_market_pulse",
        "description": (
            "Returns the macro market regime based on SPY moving averages. "
            "Output includes BULLISH/CAUTION/BEARISH status, a 0-100 score, "
            "and an action recommendation (OFFENSE/BALANCED/DEFENSE)."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_risk_audit",
        "description": (
            "Audits a portfolio of stocks for drawdown risk and alert thresholds. "
            "For each ticker, returns current price, max drawdown from recent high, "
            "and a risk alert if drawdown exceeds 10%."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "string",
                    "description": "Comma-separated list of tickers to audit, e.g. 'AAPL,TSLA,MSFT'."
                }
            },
            "required": ["tickers"]
        }
    },
    # ──────────────────────────────────────────────────────────────────────────
    # 🚀 EXPANSION ZONE: Add new alpha-mcp tools below as the server grows.
    # The data generator reads this list automatically.
    # ──────────────────────────────────────────────────────────────────────────
]
