"""Qwen chat template formatting and loss masking."""
import torch
from transformers import AutoTokenizer

_TOKENIZER_PATH = None
_TOKENIZER = None

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from pathlib import Path
        model_path = Path(__file__).parent.parent / "models" / "Qwen2.5-0.5B"
        _TOKENIZER = AutoTokenizer.from_pretrained(str(model_path))
    return _TOKENIZER


def format_messages(messages: list[dict]) -> str:
    """Apply Qwen chat template to a list of messages."""
    parts = []
    has_system = any(m["role"] == "system" for m in messages)
    if not has_system:
        parts.append(f"{IM_START}system\nYou are a helpful assistant.{IM_END}\n")
    for m in messages:
        parts.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    return "".join(parts)


def build_labels(input_ids: torch.Tensor, messages: list[dict]) -> torch.Tensor:
    """Build labels with -100 for non-assistant tokens (loss masking)."""
    tok = _get_tokenizer()
    labels = torch.full_like(input_ids, -100)

    has_system = any(m["role"] == "system" for m in messages)
    if not has_system:
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages

    full_text = format_messages([m for m in messages if m["role"] != "system"])
    if not has_system:
        full_text = f"{IM_START}system\nYou are a helpful assistant.{IM_END}\n" + full_text

    current_pos = 0
    for m in messages:
        turn_text = f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
        turn_ids = tok.encode(turn_text, add_special_tokens=False)
        turn_len = len(turn_ids)

        if m["role"] == "assistant":
            header = f"{IM_START}{m['role']}\n"
            header_ids = tok.encode(header, add_special_tokens=False)
            header_len = len(header_ids)
            content_start = current_pos + header_len
            content_end = current_pos + turn_len
            content_end = min(content_end, len(input_ids))
            content_start = min(content_start, len(input_ids))
            if content_start < content_end:
                labels[content_start:content_end] = input_ids[content_start:content_end]

        current_pos += turn_len

    return labels
