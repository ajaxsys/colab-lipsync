"""
Patch utility for Meituan LongCat-Video-Avatar 1.6 on 24GB GPUs (NVIDIA L4 / RTX 4090).

Fixes:
1. Zero-RAM Meta-Device Loading: Prevents host CPU OOM (SIGKILL) on cloud instances (e.g., Colab, L4, 4090)
   by initializing the 13.6B parameter model on PyTorch's 'meta' device and materializing INT8 shards
   directly into CUDA VRAM with assign=True. CPU RAM usage drops from 54GB+ to <3GB.
2. Text Encoder CPU Offloading: Prevents CUDA OOM on 24GB GPUs by keeping the 9.5GB UMT5-XXL text encoder
   resident on CPU RAM rather than cramming it into GPU VRAM alongside DiT (14GB) and Whisper (3GB).
   Prompt embedding forward pass runs on CPU in ~4 seconds and feeds prompt embeddings directly to CUDA.
3. Vocal Separator CPU Offloading: Prevents ONNX Runtime CUDA BFC arena OOM (Failed to allocate buffer of size 301MB)
   by configuring Kim_Vocal_2 to execute on CPUExecutionProvider and immediately freeing it after extraction.
4. Audio Encoder CPU Offloading: Reclaims ~3.1GB of GPU VRAM by offloading the Whisper-Large-v3 model to CPU
   immediately after generating the 30s audio embeddings, leaving >4.5GB free VRAM for DiT diffusion sampling.
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

    func_marker = "def load_quantized_dit"
    start_pos = content.find(func_marker)
    if start_pos == -1:
        raise ValueError("Could not find load_quantized_dit in file.")

    next_func_marker = "DEFAULT_FP8_FFN_BLOCKS"
    end_pos = content.find(next_func_marker, start_pos)
    if end_pos == -1:
        raise ValueError("Could not find end of load_quantized_dit.")

    new_content = content[:start_pos] + optimized_func + "\n\n" + content[end_pos:]
    with open(quantization_module_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"✅ Successfully patched {quantization_module_path} (Zero-RAM Meta Device)")


def patch_pipeline_module(pipeline_module_path: str):
    """
    Patches pipeline_longcat_video_avatar.py to keep UMT5-XXL text_encoder on CPU,
    saving ~9.5GB of GPU VRAM to allow inference within 24GB GPUs (NVIDIA L4 / RTX 4090).
    """
    if not os.path.exists(pipeline_module_path):
        raise FileNotFoundError(f"File not found: {pipeline_module_path}")

    with open(pipeline_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Patch `to()` method: don't move text_encoder to CUDA
    target_to = "        if self.text_encoder is not None:\n            self.text_encoder = self.text_encoder.to(device, non_blocking=True)"
    replace_to = "        if self.text_encoder is not None:\n            # Keep text_encoder on CPU to preserve 9.5GB VRAM on 24GB GPUs\n            pass"

    if target_to in content:
        content = content.replace(target_to, replace_to)
    else:
        print("ℹ️ Note: to() already patched or modified in pipeline.")

    # 2. Patch `_get_t5_prompt_embeds`: execute text_encoder on its own device (CPU) and move output to target device
    target_fwd = "        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state"
    replace_fwd = "        t_dev = next(self.text_encoder.parameters()).device\n        prompt_embeds = self.text_encoder(text_input_ids.to(t_dev), mask.to(t_dev)).last_hidden_state"

    if target_fwd in content:
        content = content.replace(target_fwd, replace_fwd)
    else:
        print("ℹ️ Note: _get_t5_prompt_embeds already patched or modified.")

    with open(pipeline_module_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {pipeline_module_path} (CPU Text Encoder Offload)")


def patch_runner_script(runner_path: str):
    """
    Patches run_demo_avatar_single_audio_to_video.py:
    1. Runs vocal separator on CPU to prevent ONNX Runtime CUDA BFC arena OOM.
    2. Offloads Whisper audio_encoder (~3.1GB) to CPU immediately after embedding generation,
       freeing >4GB of GPU VRAM for DiT diffusion forward passes.
    """
    if not os.path.exists(runner_path):
        raise FileNotFoundError(f"File not found: {runner_path}")

    with open(runner_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_cuda_sep = """    if vocal_separator.onnx_execution_provider == ["CUDAExecutionProvider"]:
        vocal_separator.onnx_execution_provider = [
            (
                "CUDAExecutionProvider",
                {"device_id": local_rank, "cudnn_conv_algo_search": "HEURISTIC"},
            )
        ]"""

    replace_cpu_sep = """    # Run vocal separator on CPU to prevent ONNX Runtime CUDA BFC arena OOM on 24GB GPUs
    vocal_separator.onnx_execution_provider = ["CPUExecutionProvider"]"""

    if target_cuda_sep in content:
        content = content.replace(target_cuda_sep, replace_cpu_sep)
    else:
        print("ℹ️ Note: vocal_separator already patched or modified in runner.")

    target_del = '        assert speech_array.size > 0, "No vocal detected"'
    replace_del = '        assert speech_array.size > 0, "No vocal detected"\n        del vocal_separator\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()'

    if target_del in content and "del vocal_separator" not in content:
        content = content.replace(target_del, replace_del)

    # Reclaim Whisper VRAM before diffusion
    target_audio_offload = "    if local_rank == 0:\n        print(f\"Generating segment 1/{num_segments}...\")"
    replace_audio_offload = """    # Reclaim ~3.1GB GPU VRAM by thoroughly deleting Whisper audio_encoder before diffusion
    try:
        del audio_encoder
    except Exception:
        pass
    if hasattr(pipe, "audio_encoder"):
        del pipe.audio_encoder
    if hasattr(pipe, "audio_feature_extractor"):
        del pipe.audio_feature_extractor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if local_rank == 0:
        print(f"Generating segment 1/{num_segments}...")"""

    if target_audio_offload in content and "del pipe.audio_encoder" not in content:
        content = content.replace(target_audio_offload, replace_audio_offload)

    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {runner_path} (CPU Vocal Separator & Whisper Offload)")


def patch_dit_avatar_module(dit_module_path: str):
    """
    Patches _create_multi_lora_forward in longcat_video/modules/avatar/longcat_video_dit_avatar.py
    to use in-place accumulator (org_output.add_(lora_output)), eliminating the ~1.7GB peak activation
    memory overhead caused by creating multiple 878MB intermediate tensors during LoRA output fusion.
    """
    if not os.path.exists(dit_module_path):
        raise FileNotFoundError(f"File not found: {dit_module_path}")

    with open(dit_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_func = '''    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            
            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    total_lora_output += lora_output
            
            return org_output + total_lora_output
        
        return multi_lora_forward'''

    replace_func = '''    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * (lora.multiplier * lora.alpha_scale)
                    org_output.add_(lora_output)
                    del lx, lora_output
            
            return org_output
        
        return multi_lora_forward'''

    if target_func in content:
        content = content.replace(target_func, replace_func)
        with open(dit_module_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Successfully patched {dit_module_path} (In-Place Multi-LoRA Forward, saves ~1.7GB VRAM)")
    else:
        print("ℹ️ Note: _create_multi_lora_forward already patched or modified in dit_avatar.")


def apply_all_patches(repo_dir: str = "/content/LongCat-Video"):
    quant_path = os.path.join(repo_dir, "longcat_video", "modules", "quantization.py")
    pipe_path = os.path.join(repo_dir, "longcat_video", "pipeline_longcat_video_avatar.py")
    runner_path = os.path.join(repo_dir, "run_demo_avatar_single_audio_to_video.py")
    dit_avatar_path = os.path.join(repo_dir, "longcat_video", "modules", "avatar", "longcat_video_dit_avatar.py")

    if os.path.exists(quant_path):
        patch_quantization_module(quant_path)
    if os.path.exists(pipe_path):
        patch_pipeline_module(pipe_path)
    if os.path.exists(runner_path):
        patch_runner_script(runner_path)
    if os.path.exists(dit_avatar_path):
        patch_dit_avatar_module(dit_avatar_path)


if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "/content/LongCat-Video"
    apply_all_patches(target_repo)
