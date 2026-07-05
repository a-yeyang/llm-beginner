"""DPO training: preference alignment on top of SFT checkpoint."""
import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from src.lora import inject_lora, LoRALinear
from src.chat import format_messages
from train_sft import load_lora_weights, save_lora_weights, collate_fn

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"


def get_log_probs(model, input_ids, attention_mask, labels):
    """Compute per-token log probs for the labeled (non -100) positions."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target = labels[:, 1:]
    mask = (target != -100).float()
    safe_target = target.clamp(min=0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, safe_target.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    reward_margin = (chosen_rewards - rejected_rewards).detach().mean()
    return loss, reward_margin


class DPODataset(Dataset):
    def __init__(self, tokenizer, max_length=512, max_samples=5000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print("Loading DPO preference data (hiyouga/DPO-En-Zh-20k, zh) ...")
        ds = load_dataset("hiyouga/DPO-En-Zh-20k", "zh", split="train")
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            convs = item.get("conversations", [])
            prompt = ""
            for turn in convs:
                if turn.get("from") == "human":
                    prompt = turn["value"]
                    break
            chosen_obj = item.get("chosen", {})
            rejected_obj = item.get("rejected", {})
            chosen = chosen_obj.get("value", "") if isinstance(chosen_obj, dict) else str(chosen_obj)
            rejected = rejected_obj.get("value", "") if isinstance(rejected_obj, dict) else str(rejected_obj)
            if prompt and chosen and rejected:
                self.samples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })
        print(f"Loaded {len(self.samples)} preference pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        chosen_msgs = [
            {"role": "user", "content": s["prompt"]},
            {"role": "assistant", "content": s["chosen"]},
        ]
        rejected_msgs = [
            {"role": "user", "content": s["prompt"]},
            {"role": "assistant", "content": s["rejected"]},
        ]
        chosen_text = format_messages(chosen_msgs)
        rejected_text = format_messages(rejected_msgs)

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )

        chosen_ids = chosen_enc.input_ids[0]
        rejected_ids = rejected_enc.input_ids[0]

        from src.chat import build_labels
        chosen_labels = build_labels(chosen_ids, chosen_msgs)
        rejected_labels = build_labels(rejected_ids, rejected_msgs)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_enc.attention_mask[0],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_enc.attention_mask[0],
            "rejected_labels": rejected_labels,
        }


def dpo_collate_fn(batch, pad_token_id=0):
    def pad_stack(items, pad_val):
        max_len = max(x.size(0) for x in items)
        return torch.stack([
            torch.cat([x, torch.full((max_len - x.size(0),), pad_val,
                                     dtype=x.dtype)])
            for x in items
        ])

    return {
        "chosen_input_ids": pad_stack([b["chosen_input_ids"] for b in batch],
                                      pad_token_id),
        "chosen_attention_mask": pad_stack(
            [b["chosen_attention_mask"] for b in batch], 0),
        "chosen_labels": pad_stack([b["chosen_labels"] for b in batch], -100),
        "rejected_input_ids": pad_stack(
            [b["rejected_input_ids"] for b in batch], pad_token_id),
        "rejected_attention_mask": pad_stack(
            [b["rejected_attention_mask"] for b in batch], 0),
        "rejected_labels": pad_stack(
            [b["rejected_labels"] for b in batch], -100),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model: load base + inject LoRA + load SFT weights
    policy = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16
    )
    inject_lora(policy, target_modules=args.target_modules.split(","),
                r=args.lora_rank, alpha=args.lora_alpha)

    sft_dir = ROOT / "ckpt" / "sft"
    if sft_dir.exists() and (sft_dir / "lora_weights.pt").exists():
        load_lora_weights(policy, sft_dir)
    else:
        print("WARNING: No SFT checkpoint found, starting from base LoRA")

    # Reference model: deep copy of policy, frozen
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()

    policy.to(device)
    ref_model.to(device)

    dataset = DPODataset(tokenizer, max_length=args.max_length,
                         max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: dpo_collate_fn(b, tokenizer.pad_token_id),
        num_workers=0, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    policy.train()
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0
        total_margin = 0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            c_ids = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            c_labels = batch["chosen_labels"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            r_labels = batch["rejected_labels"].to(device)

            policy_chosen_logps = get_log_probs(policy, c_ids, c_mask, c_labels)
            policy_rejected_logps = get_log_probs(policy, r_ids, r_mask, r_labels)

            with torch.no_grad():
                ref_chosen_logps = get_log_probs(ref_model, c_ids, c_mask,
                                                 c_labels)
                ref_rejected_logps = get_log_probs(ref_model, r_ids, r_mask,
                                                   r_labels)

            loss, reward_margin = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta=args.beta
            )
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item()
            total_margin += reward_margin.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                margin=f"{reward_margin.item():.4f}"
            )

        avg_loss = total_loss / len(dataloader)
        avg_margin = total_margin / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}, "
              f"avg reward margin: {avg_margin:.4f}")

    save_dir = ROOT / "ckpt" / "dpo"
    save_lora_weights(policy, save_dir)
    print("DPO training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
