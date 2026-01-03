#!/usr/bin/env python3
"""
adapter_inspect.py

Utilities to inspect PEFT adapter (LoRA) weights in a model, compute the
effective delta W = B @ A * scaling for modules that have LoRA, list which
layers/modules are covered, rank them by size, and provide projection tests
against an SAE decoder if provided.

Usage examples:
  python scripts/adapter_inspect.py --model-name gpt... --adapter-path ./adapter --list
  python scripts/adapter_inspect.py --model-name gpt... --adapter-path ./adapter --module 'model.layers.15.self_attn.q_proj' --show-delta --save-delta delta.npz

This script keeps the PeftModel wrapper (does not merge weights) so you can
toggle adapters on/off and run dynamic activation checks.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import torch

try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover - friendly fallback message
    raise RuntimeError("peft is required to run this script (pip install peft)") from e

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError("transformers is required (pip install transformers)") from e


def find_lora_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Return a dict mapping module path -> module that contains LoRA params.

    We consider a module to be LoRA-enabled if it has attributes named
    'lora_A' and 'lora_B' (common convention used by PEFT) or if named children
    contain those attributes.
    """
    lora_modules = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            lora_modules[name] = mod
        else:
            # sometimes lora_A/B are stored as dict attributes under the module
            # e.g., lora_A['default'] etc. We'll accept presence of either.
            la = getattr(mod, "lora_A", None)
            lb = getattr(mod, "lora_B", None)
            if la is not None and lb is not None:
                lora_modules[name] = mod
    return lora_modules


def param_to_tensor(p):
    """Given a param-like object, return a torch.Tensor of weights.

    Accepts nn.Parameter, nn.Module with .weight, or plain tensor.
    """
    if isinstance(p, torch.nn.Parameter):
        return p.data
    if isinstance(p, torch.Tensor):
        return p
    # Module with weight attribute
    w = getattr(p, "weight", None)
    if isinstance(w, torch.nn.Parameter) or isinstance(w, torch.Tensor):
        return w.data
    raise TypeError(f"Unsupported parameter type: {type(p)}")


def get_lora_delta_from_module(mod: torch.nn.Module) -> Optional[torch.Tensor]:
    """Reconstruct Delta W = B @ A * scaling for a given LoRA-enabled module.

    Returns None if the module does not contain LoRA params.
    """
    # PEFT often stores dicts keyed by 'default' or similar; handle both.
    if not hasattr(mod, "lora_A") or not hasattr(mod, "lora_B"):
        return None

    A_raw = getattr(mod, "lora_A")
    B_raw = getattr(mod, "lora_B")

    # support dict style: {'default': param}
    if isinstance(A_raw, dict):
        A_raw = next(iter(A_raw.values()))
    if isinstance(B_raw, dict):
        B_raw = next(iter(B_raw.values()))

    A = param_to_tensor(A_raw)
    B = param_to_tensor(B_raw)

    # scaling may be a dict or a float
    scaling = getattr(mod, "scaling", None)
    if isinstance(scaling, dict):
        scaling = next(iter(scaling.values()))
    if scaling is None:
        # Some LoRA implementations use alpha and r
        alpha = getattr(mod, "alpha", None)
        r = getattr(mod, "r", None)
        if alpha is not None and r is not None and r != 0:
            scaling = float(alpha) / float(r)
        else:
            scaling = 1.0

    A = A.to(dtype=torch.get_default_dtype())
    B = B.to(dtype=torch.get_default_dtype())
    delta = (B @ A) * float(scaling)
    return delta


def module_delta_stats(delta: torch.Tensor) -> Tuple[Tuple[int, ...], float]:
    shape = tuple(delta.shape)
    norm = float(delta.norm().item())
    return shape, norm


def load_sae(path: str):
    """Try to load an SAE-like object from a file.

    We accept torch saved objects that contain a decoder matrix under common
    attribute names like 'W_dec', 'decoder', or 'W_decoder'. If the saved file
    is a plain dict with a numpy array, we convert it to a tensor.
    """
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu")
    # If it's a dict-like state
    if isinstance(obj, dict):
        for key in ("W_dec", "decoder", "W_decoder", "Wdec", "W_decoded"):
            if key in obj:
                w = obj[key]
                return torch.as_tensor(w)
        # try first tensor-like value
        for v in obj.values():
            if isinstance(v, (torch.Tensor, list)):
                return torch.as_tensor(v)
    # object with attribute
    for attr in ("W_dec", "decoder", "W_decoder", "Wdec"):
        if hasattr(obj, attr):
            return torch.as_tensor(getattr(obj, attr))
    # otherwise return the object itself and let caller handle it
    return obj


def check_projection(lora_delta: torch.Tensor, sae_decoder: torch.Tensor, top_k: int = 5):
    """Project LoRA delta onto SAE decoder features and return top-k features.

    lora_delta: (d_out, d_in) or (d_model, d_model)
    sae_decoder: (n_features, d_model)
    Returns top indices and scores.
    """
    # Align shapes: ensure d_model matches
    d_model = sae_decoder.shape[1]
    if lora_delta.shape[0] != d_model and lora_delta.shape[1] != d_model:
        # try to transpose if shapes swapped
        if lora_delta.shape[1] == d_model:
            lora_delta = lora_delta.t()
        else:
            raise ValueError(f"Delta shape {lora_delta.shape} incompatible with SAE d_model={d_model}")

    # Normalize for cosine-like projection
    delta_norm = torch.nn.functional.normalize(lora_delta, dim=0)
    sae_norm = torch.nn.functional.normalize(sae_decoder, dim=1)

    projection = sae_norm @ delta_norm  # (n_features, d_model)
    feature_scores = projection.abs().sum(dim=1)
    top_scores, top_idx = torch.topk(feature_scores, k=min(top_k, feature_scores.numel()))
    return top_idx.tolist(), top_scores.tolist()


class ActivationCapture:
    def __init__(self):
        self.activation = None

    def hook(self, module, input, output):
        # Save a detached copy
        self.activation = output.detach().cpu()


def analyze_activations(peft_model, sae, tokenizer, prompt: str, target_module_path: str):
    """Run the model with adapter off and on and compare SAE feature activations.

    target_module_path should be one of the names from model.named_modules().
    Returns a dict with top-diff features and MSE reconstruction.
    """
    device = next(peft_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    module_map = dict(peft_model.named_modules())
    if target_module_path not in module_map:
        raise KeyError(f"Module path '{target_module_path}' not found in model modules")
    target_module = module_map[target_module_path]

    capture = ActivationCapture()
    handle = target_module.register_forward_hook(capture.hook)

    # Run with adapter disabled
    with peft_model.disable_adapter():
        with torch.no_grad():
            _ = peft_model(**inputs)
    base_act = capture.activation.clone()

    # Run with adapter enabled
    with torch.no_grad():
        _ = peft_model(**inputs)
    ft_act = capture.activation.clone()
    handle.remove()

    # Flatten batch/time dims for SAE
    base_flat = base_act.flatten(0, 1)
    ft_flat = ft_act.flatten(0, 1)

    # SAE expected interface: encode(tensor) -> features tensor (N, n_features)
    # We allow SAE to be a plain decoder matrix too.
    if hasattr(sae, "encode"):
        base_feats = sae.encode(base_flat)
        ft_feats = sae.encode(ft_flat)
    else:
        # assume sae is decoder matrix W_dec (n_features, d_model)
        W = sae
        base_feats = base_flat @ W.t()
        ft_feats = ft_flat @ W.t()

    diff = ft_feats - base_feats
    boosted = diff.sum(dim=0)
    top_vals, top_idx = torch.topk(boosted.abs(), k=min(5, boosted.numel()))

    # Reconstruction MSE: use base SAE decoder to reconstruct ft activations
    if hasattr(sae, "decode"):
        recon = sae.decode(ft_feats)
    else:
        recon = ft_feats @ W
    mse = (ft_flat - recon).pow(2).mean().item()

    return {"top_diff_features": top_idx.tolist(), "top_diff_scores": top_vals.tolist(), "mse": mse}

def analyze_attention_sae_lora(model, sae, layer_idx):
    """
    Analyzes LoRA using an SAE trained on the 'z' vector (Concatenated Head Outputs).
    """
    # 1. Get Layer and Dimensions
    layer = model.base_model.model.model.layers[layer_idx]
    self_attn = layer.self_attn
    
    # Gemma 2 specifics
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    # The SAE input dimension should be num_heads * head_dim
    
    results = {}

    # --- Helper: Get LoRA Delta ---
    def get_lora_delta(module):
        if not hasattr(module, "lora_A"): return None
        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight
        scale = module.scaling["default"]
        return (B @ A) * scale

    # ==================================================
    # ANALYSIS 1: The Value (V) LoRA
    # "What raw messages is the LoRA inserting into the heads?"
    # ==================================================
    delta_V = get_lora_delta(self_attn.v_proj)
    # Shape: (num_kv_heads * head_dim, d_model)
    
    if delta_V is not None:
        # Note: Gemma 2 uses GQA (Grouped Query Attention).
        # The SAE expects (num_heads * head_dim), but delta_V corresponds to (num_kv_heads * head_dim).
        # We must repeat the delta_V rows to match the full 'z' vector space.
        
        num_kv_heads = model.config.num_key_value_heads
        num_groups = num_heads // num_kv_heads
        
        # Reshape to (num_kv_heads, head_dim, d_model)
        delta_V_reshaped = delta_V.reshape(num_kv_heads, head_dim, -1)
        
        # Repeat for each group to simulate the full 'z' vector construction
        # Shape: (num_kv_heads, num_groups, head_dim, d_model) -> (num_heads, head_dim, d_model)
        delta_V_expanded = delta_V_reshaped.unsqueeze(1).expand(-1, num_groups, -1, -1)
        delta_V_full = delta_V_expanded.reshape(num_heads * head_dim, -1)
        
        # Now we project this "Virtual Z Update" onto the SAE
        # We want to know: "If this LoRA activates, what Attn SAE features light up?"
        # SAE Decoder: (n_features, z_dim)
        # Delta V Full: (z_dim, d_model_in)
        
        # Project columns (input directions) or rows (output directions)?
        # V writes TO 'z'. So we project the COLUMNS of delta_V (the output side).
        # We sum over the input dimension (d_model) to get the "general vibe".
        
        # Normalize
        sae_dec_norm = torch.nn.functional.normalize(sae.W_dec, dim=1)
        delta_v_norm = torch.nn.functional.normalize(delta_V_full, dim=0)
        
        projection = sae_dec_norm @ delta_v_norm # (n_features, d_model)
        scores = projection.sum(dim=1)
        
        top_scores, top_feats = torch.topk(scores, k=5)
        results['v_write_features'] = top_feats.tolist()

    # ==================================================
    # ANALYSIS 2: The Output (O) LoRA
    # "How is the LoRA changing how head messages are read?"
    # ==================================================
    delta_O = get_lora_delta(self_attn.o_proj)
    # Shape: (d_model, num_heads * head_dim)
    
    if delta_O is not None:
        # O reads FROM 'z'. So we project the ROWS of delta_O (the input side).
        # This tells us: "Which SAE features (head messages) does this LoRA listen to more?"
        
        delta_o_input_dirs = delta_O.T # (z_dim, d_model)
        
        # Normalize
        delta_o_norm = torch.nn.functional.normalize(delta_o_input_dirs, dim=0)
        sae_dec_norm = torch.nn.functional.normalize(sae.W_dec, dim=1)

        # Dot product
        projection = sae_dec_norm @ delta_o_norm
        scores = projection.sum(dim=1)
        
        top_scores, top_feats = torch.topk(scores, k=5)
        results['o_read_features'] = top_feats.tolist()

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, required=True, help="Base model name or path for AutoModelForCausalLM")
    p.add_argument("--adapter-path", type=str, required=True, help="Path to PEFT adapter (do not merge)")
    p.add_argument("--adapter-file-only", action="store_true", help="Only inspect the adapter file on disk (safetensors or pt) without loading the base model")
    p.add_argument("--device", type=str, default="auto", help="Device spec for transformers (e.g., cpu or auto)")
    p.add_argument("--list", action="store_true", help="List LoRA-enabled modules and stats")
    p.add_argument("--show-largest", type=int, default=10, help="Show top-N largest LoRA deltas by norm")
    p.add_argument("--module", type=str, help="Specific module path to inspect (as in model.named_modules())")
    p.add_argument("--save-delta", type=str, help="Save the delta matrix for the specified module to an .npz file")
    p.add_argument("--sae-path", type=str, help="Optional path to SAE file to run projection/activation checks")
    p.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt for activation comparison")
    args = p.parse_args()

    if args.adapter_file_only:
        # Fast-path: read the adapter file itself (safetensors or torch) and list keys/shapes/norms
        adapter_fp = args.adapter_path
        if not os.path.exists(adapter_fp):
            raise FileNotFoundError(adapter_fp)
        print(f"Inspecting adapter file: {adapter_fp}")
        data = None
        if adapter_fp.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as load_safetensors

                data = load_safetensors(adapter_fp)
            except Exception:
                raise RuntimeError("safetensors not installed or failed to load file. Install safetensors to use this mode.")
        else:
            data = torch.load(adapter_fp, map_location="cpu")

        # data may be a dict of tensors or a nested dict
        if isinstance(data, dict):
            # flatten possible nested dicts
            flat = {}
            def walk(prefix, d):
                if isinstance(d, dict):
                    for k, v in d.items():
                        walk(prefix + ("." if prefix else "") + k, v)
                else:
                    flat[prefix] = d
            walk("", data)
            items = []
            for k, v in flat.items():
                if isinstance(v, torch.Tensor):
                    norm = float(v.norm().item())
                    items.append((k, tuple(v.shape), norm))
            items_sorted = sorted(items, key=lambda x: x[2], reverse=True)
            print(f"Found {len(items_sorted)} tensors in adapter file. Top {args.show_largest} by norm:")
            for name, shape, norm in items_sorted[:args.show_largest]:
                print(f"  {name}: shape={shape}, norm={norm:.4f}")
            return
        else:
            print("Adapter file loaded but format not recognized for fast inspection.")
            return

    # Load base model and tokenizer
    device_map = "auto" if args.device == "auto" else {"": args.device}
    print(f"Loading base model {args.model_name} on device {args.device}...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print(f"Loading adapter from {args.adapter_path} (PEFT wrapper, not merged)...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
    peft_model.eval()

    # Find LoRA modules
    lora_modules = find_lora_modules(peft_model)
    print(f"Found {len(lora_modules)} LoRA-enabled modules in the model.")

    stats = []
    for name, mod in lora_modules.items():
        delta = get_lora_delta_from_module(mod)
        if delta is None:
            continue
        shape, norm = module_delta_stats(delta)
        stats.append((name, shape, norm))

    # Sort by norm descending
    stats_sorted = sorted(stats, key=lambda x: x[2], reverse=True)

    if args.list:
        for name, shape, norm in stats_sorted:
            print(f"{name}: shape={shape}, fro_norm={norm:.4f}")

    if args.show_largest and args.show_largest > 0:  # type: ignore[name-defined]
        n = args.show_largest
        print(f"\nTop {n} modules by delta norm:")
        for name, shape, norm in stats_sorted[:n]:
            print(f"  {name}: norm={norm:.4f} shape={shape}")

    if args.module:
        if args.module not in lora_modules:
            print(f"Module '{args.module}' not found among LoRA modules. Available modules (showing first 20):")
            for nm in list(lora_modules.keys())[:20]:
                print(" ", nm)
            return
        mod = lora_modules[args.module]
        delta = get_lora_delta_from_module(mod)
        if delta is None:
            print("No delta could be computed for this module.")
            return
        print(f"Delta shape: {tuple(delta.shape)}, norm={float(delta.norm()):.4f}")
        if args.save_delta:
            import numpy as _np

            os.makedirs(os.path.dirname(args.save_delta) or ".", exist_ok=True)
            _np.savez(args.save_delta, delta=delta.cpu().numpy())
            print(f"Saved delta to {args.save_delta}")

        if args.sae_path:
            sae = load_sae(args.sae_path)
            if isinstance(sae, torch.Tensor):
                top_idx, top_scores = check_projection(delta, sae, top_k=10)
                print("Top SAE features aligned with this LoRA delta:")
                for i, s in zip(top_idx, top_scores):
                    print(f"  feature {i}: score={s:.6f}")
            else:
                print("Loaded SAE object is not a plain decoder matrix. Skipping static projection check.")
            # Optionally run activations if SAE has encode/decode and module path is exact
            if hasattr(sae, "encode"):
                print("Running activation comparison (this will run the model twice: adapter off/on).")
                res = analyze_activations(peft_model, sae, tokenizer, args.prompt, args.module)
                print("Activation analysis:")
                print(res)


if __name__ == "__main__":
    main()
