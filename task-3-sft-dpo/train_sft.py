"""SFT training on MOSS-003-sft data with hand-written LoRA."""
import json
import re
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.lora import inject_lora, LoRALinear
from src.chat import format_messages, build_labels

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"


def parse_moss_conversation(item: dict) -> list[dict]:
    """Parse a MOSS dataset item into a list of {role, content} messages."""
    messages = []
    chat = item["chat"]
    num_turns = item.get("num_turns", len(chat))

    for i in range(1, num_turns + 1):
        turn_key = f"turn_{i}"
        if turn_key not in chat:
            break
        turn = chat[turn_key]

        human_text = turn.get("Human", "")
        human_text = re.sub(r"<\|Human\|>:\s*", "", human_text)
        human_text = re.sub(r"<eoh>\s*$", "", human_text).strip()

        moss_text = turn.get("MOSS", "")
        moss_text = re.sub(r"<\|MOSS\|>:\s*", "", moss_text)
        moss_text = re.sub(r"<eom>\s*$", "", moss_text).strip()

        if human_text:
            messages.append({"role": "user", "content": human_text})
        if moss_text:
            messages.append({"role": "assistant", "content": moss_text})

    return messages


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512,
                 max_samples: int = 10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print(f"Loading data from {data_path} ...")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line)
                messages = parse_moss_conversation(item)
                if len(messages) >= 2:
                    self.samples.append(messages)
        print(f"Loaded {len(self.samples)} conversations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        text = format_messages(messages)
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        labels = build_labels(input_ids, messages)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([b["input_ids"],
                         torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"],
                             torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"],
                      torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def save_lora_weights(model, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
            state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()
            state_dict[f"{name}.r"] = module.r
            state_dict[f"{name}.scaling"] = module.scaling
    torch.save(state_dict, save_dir / "lora_weights.pt")
    print(f"LoRA weights saved to {save_dir}")


def load_lora_weights(model, load_dir: Path):
    state_dict = torch.load(load_dir / "lora_weights.pt", map_location="cpu",
                            weights_only=True)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state_dict:
                module.lora_A.data.copy_(state_dict[a_key])
                module.lora_B.data.copy_(state_dict[b_key])
    print(f"LoRA weights loaded from {load_dir}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16
    )
    inject_lora(model, target_modules=args.target_modules.split(","),
                r=args.lora_rank, alpha=args.lora_alpha)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.4%})")

    dataset = SFTDataset(args.data_path, tokenizer,
                         max_length=args.max_length,
                         max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    total_steps = len(dataloader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += outputs.loss.item()
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % 500 == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    save_dir = ROOT / "ckpt" / "sft"
    save_lora_weights(model, save_dir)
    print("SFT training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default=str(ROOT / "data" / "moss-sft" /
                                    "moss-003-sft-no-tools.jsonl"))
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
