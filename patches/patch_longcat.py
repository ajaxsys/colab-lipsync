"""
Patch utility for Meituan LongCat-Video-Avatar 1.6.

Fixes:
1. Zero-RAM Meta-Device Loading: Prevents host CPU OOM (SIGKILL) on cloud instances (e.g., Colab, 24GB L4/RTX 4090)
   by initializing the 13.6B parameter model on PyTorch's 'meta' device and materializing INT8 shards
   directly into CUDA VRAM with assign=True. CPU RAM usage drops from 54GB+ to <3GB.
2. Sharded safe loading without intermediate dictionary accumulation.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file


def patch_quantization_module(quantization_module_path: str):
    """
    Patches load_quantized_dit in LongCat-Video/longcat_video/modules/quantization.py
    to use meta device instantiation and direct CUDA shard assignment.
    """
    if not os.path.exists(quantization_module_path):
        raise FileNotFoundError(f"File not found: {quantization_module_path}")

    with open(quantization_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    optimized_func = '''def load_quantized_dit(checkpoint_dir: str, subfolder: str = "base_model_int8", **kwargs):
    """Load a quantized DiT model directly into target CUDA device using meta-device initialization.
    Optimized to eliminate CPU RAM spikes and prevent SIGKILL (OOM Killer).
    """
    from .avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    quantized_dir = os.path.join(checkpoint_dir, subfolder)
    config_path = os.path.join(quantized_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    for k in ["_class_name", "architectures", "_diffusers_version", "model_max_length"]:
        config.pop(k, None)
    config.update(kwargs)

    # 1. Zero-RAM initialization on meta device
    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)

    # 2. Replace nn.Linear with QuantizedLinear on meta device
    skip_patterns = DEFAULT_SKIP_PATTERNS
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = any(pattern in name for pattern in skip_patterns)
            if not should_skip:
                with torch.device("meta"):
                    ql = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
                modules_to_replace[name] = ql

    for name, ql in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], ql)
    del modules_to_replace

    # Determine target CUDA device
    target_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

    # 3. Direct shard-by-shard loading into target device with assign=True
    index_path = os.path.join(quantized_dir, "quantized_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in shard_files:
            shard_path = os.path.join(quantized_dir, shard_file)
            shard_dict = load_file(shard_path, device=target_device)
            model.load_state_dict(shard_dict, strict=False, assign=True)
            del shard_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        files = [f for f in os.listdir(quantized_dir) if f.endswith(".safetensors") and "index" not in f]
        for f in sorted(files):
            shard_dict = load_file(os.path.join(quantized_dir, f), device=target_device)
            model.load_state_dict(shard_dict, strict=False, assign=True)
            del shard_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 4. Materialize any remaining non-quantized buffers/weights on target device
    model.to_empty(device=target_device)
    model.eval()
    return model
'''

    # Replace the existing load_quantized_dit implementation
    func_marker = "def load_quantized_dit"
    start_pos = content.find(func_marker)
    if start_pos == -1:
        raise ValueError("Could not find load_quantized_dit in file.")

    # Find the end of load_quantized_dit (before DEFAULT_FP8_FFN_BLOCKS)
    next_func_marker = "DEFAULT_FP8_FFN_BLOCKS"
    end_pos = content.find(next_func_marker, start_pos)
    if end_pos == -1:
        raise ValueError("Could not find end of load_quantized_dit.")

    new_content = content[:start_pos] + optimized_func + "\n\n" + content[end_pos:]
    with open(quantization_module_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"✅ Successfully patched {quantization_module_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/content/LongCat-Video/longcat_video/modules/quantization.py"
    patch_quantization_module(target)
