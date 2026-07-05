"""Compare base model vs SFT vs DPO outputs on sample prompts."""
import argparse
from pathlib import Path

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lora import inject_lora
from src.chat import format_messages
from train_sft import load_lora_weights

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"

PROMPTS = [
    "请介绍一下什么是深度学习。",
    "如何学习编程？",
    "帮我写一首关于春天的诗。",
]


def generate(model, tokenizer, prompt_text, max_new_tokens=200):
    msgs = [{"role": "user", "content": prompt_text}]
    text = format_messages(msgs)
    text += "<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True)
    return generated.strip()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+",
                        default=["base", "sft", "dpo"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for ckpt_name in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"  Model: {ckpt_name}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH), dtype=torch.bfloat16
        )

        if ckpt_name in ("sft", "dpo"):
            inject_lora(model, target_modules=["q_proj", "v_proj"],
                        r=8, alpha=16)
            ckpt_dir = ROOT / "ckpt" / ckpt_name
            if (ckpt_dir / "lora_weights.pt").exists():
                load_lora_weights(model, ckpt_dir)
            else:
                print(f"  [SKIP] {ckpt_dir} not found")
                continue

        model.to(device).eval()

        for prompt in PROMPTS:
            print(f"\n  Q: {prompt}")
            answer = generate(model, tokenizer, prompt)
            print(f"  A: {answer}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
