"""Hand-written LoRA: low-rank adaptation for linear layers."""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with low-rank A/B matrices."""

    def __init__(self, original: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.original = original
        self.r = r
        self.scaling = alpha / r

        in_features = original.in_features
        out_features = original.out_features

        dtype = original.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(in_features, r, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (x @ self.lora_A) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, target_modules: list[str],
                r: int = 8, alpha: float = 16) -> nn.Module:
    """Replace target linear layers with LoRA-wrapped versions."""
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                lora_layer = LoRALinear(module, r=r, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA weights back into the original linear layers."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = (module.lora_B @ module.lora_A.T) * module.scaling
                module.original.weight.add_(delta)
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], module.original)
    return model
