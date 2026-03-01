"""
generate.py — Load a saved JEPALoRAModel and run generation.

Usage
-----
python generate.py \
    --checkpoint ./checkpoints/qwen25-0.5b-jepa \
    --prompt "The key insight about large language models is" \
    --max_new_tokens 200 \
    --temperature 0.7

The script merges LoRA weights into the base model before generation
so there is no runtime overhead from the adapter.
"""

import argparse
import sys
import torch
from pathlib import Path

from model import JEPALoRAModel


def generate(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Loading checkpoint from {args.checkpoint} on {device}")
    model = JEPALoRAModel.load(args.checkpoint, device=device)

    tok = model.tokenizer
    inputs = tok(args.prompt, return_tensors="pt").to(device)

    # Merge LoRA weights into base for faster inference
    merged = model.backbone.merge_and_unload()
    merged.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature if args.temperature > 0 else 1.0,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    with torch.no_grad():
        output_ids = merged.generate(**inputs, **gen_kwargs)

    # Only print the newly generated tokens (after the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][prompt_len:]
    generated = tok.decode(new_ids, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"PROMPT:\n{args.prompt}")
    print(f"\nGENERATED:\n{generated}")
    print(f"{'='*60}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with a saved JEPALoRAModel")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to saved checkpoint directory")
    p.add_argument("--prompt", type=str,
                   default="The key insight about large language models is")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0 = greedy decoding")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu / mps — auto-detected if omitted")
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
