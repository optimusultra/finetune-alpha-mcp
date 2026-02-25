"""
Alpha-MCP Dispatcher Micro-Script
==================================
A high-speed bridge between natural language and Alpha-MCP tool calls.
Designed to be called by OpenClaw Skills.

INPUT:  Query string (first argument)
OUTPUT: JSON tool call block
"""

import sys
import json
import torch
import os

# Suppress warnings for clean stdout
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No query provided"}))
        sys.exit(1)

    query = sys.argv[1]
    
    # Paths (Constants relative to this package)
    BASE_MODEL = "google/functiongemma-270m-it"
    ADAPTER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs/alpha_functiongemma")

    # Check if adapter exists
    if not os.path.exists(ADAPTER_PATH):
        print(json.dumps({"error": "Fine-tuned adapter not found at " + ADAPTER_PATH}))
        sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()

        # Format prompt precisely as trained
        prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=False, # Deterministic for tool calling
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_resp = tokenizer.decode(out[0], skip_special_tokens=False)
        
        # Extract the <start_function_call> block
        if "<start_function_call>" in full_resp:
            call_block = full_resp.split("<start_function_call>")[1].split("<end_function_call>")[0]
            # Form: call:name{"arg": "val"}
            name = call_block.split("call:")[1].split("{")[0]
            args_str = "{" + call_block.split("{", 1)[1]
            
            # Final structured output for the Skill
            print(json.dumps({
                "tool": name,
                "arguments": json.loads(args_str),
                "raw": call_block
            }, indent=2))
        else:
            print(json.dumps({"error": "No tool call generated", "raw": full_resp}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
