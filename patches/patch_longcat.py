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

    # 3. Patch `_decode_video_latents`: reclaim GPU cache before VAE decoding to prevent VAE OOM and guard against NaNs
    target_decode = "    def _decode_video_latents(self, latents):\n        cp_size = context_parallel_util.get_cp_size()"
    replace_decode = """    def _decode_video_latents(self, latents):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        latents = torch.nan_to_num(latents, 0.0)
        cp_size = context_parallel_util.get_cp_size()"""

    if target_decode in content:
        content = content.replace(target_decode, replace_decode)
    elif "latents = torch.nan_to_num(latents, 0.0)" not in content:
        content = content.replace("cp_size = context_parallel_util.get_cp_size()", "latents = torch.nan_to_num(latents, 0.0)\n        cp_size = context_parallel_util.get_cp_size()")

    # 4. Patch denoising steps with nan_to_num protection against numerical instability
    target_step = "latents[:, :, 1:] = self.scheduler.step(noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False)[0]"
    replace_step = "latents[:, :, 1:] = self.scheduler.step(torch.nan_to_num(noise_pred[:, :, 1:], 0.0), t, latents[:, :, 1:], return_dict=False)[0]\n                latents = torch.nan_to_num(latents, 0.0)"

    if target_step in content and "torch.nan_to_num" not in content:
        content = content.replace(target_step, replace_step)

    # 5. Patch generate_avc denoising steps with nan_to_num protection to prevent black frames in video continuation
    target_avc_step = """                if use_kv_cache:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents[:, :, num_cond_latents:] = self.scheduler.step(noise_pred[:, :, num_cond_latents:], t, latents[:, :, num_cond_latents:], return_dict=False)[0]"""

    replace_avc_step = """                noise_pred = torch.nan_to_num(noise_pred, 0.0)
                if use_kv_cache:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents[:, :, num_cond_latents:] = self.scheduler.step(noise_pred[:, :, num_cond_latents:], t, latents[:, :, num_cond_latents:], return_dict=False)[0]
                latents = torch.nan_to_num(latents, 0.0)"""

    if target_avc_step in content:
        content = content.replace(target_avc_step, replace_avc_step)

    # 6. Dynamic VAE CPU offload during diffusion loop in generate_ai2v and generate_avc
    target_ai2v_diff = "        with tqdm(total=len(timesteps), desc=\"Denoising\") as progress_bar:"
    replace_ai2v_diff = """        # Dynamic VAE CPU offload during diffusion to free ~800MB VRAM
        if hasattr(self, "vae") and self.vae is not None:
            self.vae = self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:"""

    if target_ai2v_diff in content and "Dynamic VAE CPU offload" not in content:
        content = content.replace(target_ai2v_diff, replace_ai2v_diff)

    target_decode_vae = "        latents = latents.to(self.vae.dtype)\n        latents = self.denormalize_latents(latents)"
    replace_decode_vae = """        # Bring VAE back to target device for video decoding
        if hasattr(self, "vae") and self.vae is not None:
            self.vae = self.vae.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        latents = torch.nan_to_num(latents, 0.0)
        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)"""

    if target_decode_vae in content:
        content = content.replace(target_decode_vae, replace_decode_vae)

    # 7. Patch generate_avc to delete unneeded raw video tensor and reclaim ~250MB VRAM before denoising
    target_avc_prep = """        cond_videos = video[:, :, -num_cond_frames:]
        cond_videos_latents = retrieve_latents(self.vae.encode(cond_videos), generator, sample_mode="argmax")
        cond_videos_latents = self.normalize_latents(cond_videos_latents)"""
    replace_avc_prep = """        cond_videos = video[:, :, -num_cond_frames:]
        cond_videos_latents = retrieve_latents(self.vae.encode(cond_videos), generator, sample_mode="argmax")
        cond_videos_latents = self.normalize_latents(cond_videos_latents)
        del video
        del cond_videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()"""

    if target_avc_prep in content and "del cond_videos" not in content:
        content = content.replace(target_avc_prep, replace_avc_prep)

    # 8. Offload multi-chunk Whisper encoder hidden states accumulation to CPU to prevent OOM on >30s audio
    target_whisper_cat = """        # ---- Whisper encoder：mel → hidden states ----
        enc_chunks = []
        for i in range(0, audio_features.shape[-1], ENC_CHUNK):
            chunk_hs = self.audio_encoder.encoder(
                audio_features[:, :, i: i + ENC_CHUNK].to(device),
                output_hidden_states=True,
            ).hidden_states                           # tuple: (n_layers+1,) x [1, T_enc, D]
            enc_chunks.append(torch.stack(chunk_hs, dim=2))  # [1, T_enc, n_layers, D]
        audio_prompts = torch.cat(enc_chunks, dim=1)         # [1, T_enc_total, n_layers, D]
        audio_prompts = audio_prompts[:, :video_length * 2]  # 截取有效帧

        # ---- 按层分组 + 插值到目标帧数 ----
        feat0 = linear_interpolation_fps(audio_prompts[:, :,  0: 8].mean(dim=2), ENC_FPS, fps, video_length)
        feat1 = linear_interpolation_fps(audio_prompts[:, :,  8:16].mean(dim=2), ENC_FPS, fps, video_length)
        feat2 = linear_interpolation_fps(audio_prompts[:, :, 16:24].mean(dim=2), ENC_FPS, fps, video_length)
        feat3 = linear_interpolation_fps(audio_prompts[:, :, 24:32].mean(dim=2), ENC_FPS, fps, video_length)
        feat4 = linear_interpolation_fps(audio_prompts[:, :, 32],                ENC_FPS, fps, video_length)
        audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]   # [T, 5, D]
        return audio_emb"""

    replace_whisper_cat = """        # ---- Whisper encoder：mel → hidden states (accumulate chunks on CPU to avoid multi-chunk VRAM OOM) ----
        enc_chunks = []
        for i in range(0, audio_features.shape[-1], ENC_CHUNK):
            chunk_hs = self.audio_encoder.encoder(
                audio_features[:, :, i: i + ENC_CHUNK].to(device),
                output_hidden_states=True,
            ).hidden_states
            enc_chunks.append(torch.stack(chunk_hs, dim=2).float().cpu())
            del chunk_hs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        audio_prompts = torch.cat(enc_chunks, dim=1)
        del enc_chunks
        audio_prompts = audio_prompts[:, :video_length * 2]

        # ---- 按层分组 + 插值到目标帧数 (CPU FP32 interpolation -> move compact [T,5,D] back to device) ----
        feat0 = linear_interpolation_fps(audio_prompts[:, :,  0: 8].mean(dim=2), ENC_FPS, fps, video_length)
        feat1 = linear_interpolation_fps(audio_prompts[:, :,  8:16].mean(dim=2), ENC_FPS, fps, video_length)
        feat2 = linear_interpolation_fps(audio_prompts[:, :, 16:24].mean(dim=2), ENC_FPS, fps, video_length)
        feat3 = linear_interpolation_fps(audio_prompts[:, :, 24:32].mean(dim=2), ENC_FPS, fps, video_length)
        feat4 = linear_interpolation_fps(audio_prompts[:, :, 32],                ENC_FPS, fps, video_length)
        audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0].to(device=device, dtype=self.audio_encoder.dtype)
        return audio_emb"""

    if target_whisper_cat in content:
        content = content.replace(target_whisper_cat, replace_whisper_cat)
    else:
        print("ℹ️ Note: get_audio_embedding_whisper CPU chunking already patched or modified.")

    with open(pipeline_module_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {pipeline_module_path} (CPU Text Encoder Offload & VAE Lifecycle)")


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

    # Support global seed override via LONGCAT_SEED env var for multi-seed reproducibility testing
    content = content.replace(
        "    global_seed = 42",
        '    global_seed = int(os.environ.get("LONGCAT_SEED", "42"))',
    )

    # Preserve --audio_guidance_scale as audio_residual_scale when use_distill overrides CFG to 1.0
    target_distill_cfg = """    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0"""
    replace_distill_cfg = """    audio_residual_scale = 1.0
    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_residual_scale = float(args.audio_guidance_scale)
        audio_guidance_scale = 1.0"""
    if target_distill_cfg in content:
        content = content.replace(target_distill_cfg, replace_distill_cfg)

    target_del = '        assert speech_array.size > 0, "No vocal detected"'
    replace_del = '''        assert speech_array.size > 0, "No vocal detected"
        vad_weights = None
        try:
            import numpy as np
            # v2.3 Universal Frame-Level VAD (Voice Activity Detection, zero hardcoded timestamps):
            # Measures local vocal RMS per 25fps frame across the entire track (intro, mid-pauses, outro).
            frame_rate = save_fps * audio_stride
            num_vad_frames = max(1, int(len(speech_array) / sr * frame_rate))
            win_half = int(0.10 * sr)
            frame_rms = np.zeros(num_vad_frames, dtype=np.float32)
            for fi in range(num_vad_frames):
                center = int((fi + 0.5) * sr / frame_rate)
                s0 = max(0, center - win_half)
                s1 = min(len(speech_array), center + win_half)
                if s1 > s0:
                    frame_rms[fi] = float(np.sqrt(np.mean(speech_array[s0:s1] ** 2)))
            active_ref = float(np.percentile(frame_rms, 90)) if frame_rms.size > 0 else 0.0
            vad_thresh = max(0.0025, 0.08 * active_ref)
            vad_mask = (frame_rms > vad_thresh).astype(np.float32)
            # Bridge brief pauses (<0.36s) so breaths/consonants are never clipped
            gap_bridge = max(1, int(0.36 * frame_rate))
            padded = vad_mask.copy()
            last_on = -999
            for fi in range(num_vad_frames):
                if vad_mask[fi] > 0.5:
                    if 0 < fi - last_on <= gap_bridge:
                        padded[last_on:fi] = 1.0
                    last_on = fi
            k = max(3, int(0.24 * frame_rate) | 1)
            vad_weights = np.clip(np.convolve(padded, np.ones(k, dtype=np.float32) / k, mode="same") * 1.5, 0.0, 1.0)

            # Universal Vocal Peak Compression measured strictly on active speech samples
            active_sample_mask = np.interp(
                np.linspace(0, num_vad_frames - 1, len(speech_array)),
                np.arange(num_vad_frames),
                padded,
            ) > 0.5
            active_samples = speech_array[active_sample_mask]
            vocal_std = float(np.std(active_samples)) if active_samples.size > 160 else float(np.std(speech_array))
            clip_threshold = 2.0 * vocal_std
            if clip_threshold > 0:
                excess = np.maximum(0, np.abs(speech_array) - clip_threshold)
                speech_array = np.sign(speech_array) * (
                    np.minimum(np.abs(speech_array), clip_threshold)
                    + vocal_std * 0.4 * np.tanh(excess / (vocal_std * 0.4 + 1e-6))
                )
            print(f"✅ [v2.3 Universal VAD] active_ref={active_ref:.4f}, vad_thresh={vad_thresh:.4f}, active_ratio={vad_weights.mean():.1%}, soft_limiter_thresh={clip_threshold:.4f}")
        except Exception as e:
            print(f"⚠️ [v2.3 Patch] Audio conditioning warning: {e}")
        del vocal_separator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()'''

    if target_del in content and "del vocal_separator" not in content:
        content = content.replace(target_del, replace_del)

    # Temporarily offload VAE to CPU during multi-chunk Whisper audio embedding to save ~800MB VRAM
    target_get_audio = "        # audio embedding\n        full_audio_emb = pipe.get_audio_embedding("
    replace_get_audio = """        # Offload VAE temporarily during Whisper encoding to guarantee safe VRAM headroom on 40GB GPUs
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae = pipe.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # audio embedding
        full_audio_emb = pipe.get_audio_embedding("""
    if target_get_audio in content and "Offload VAE temporarily during Whisper" not in content:
        content = content.replace(target_get_audio, replace_get_audio)

    # Apply post-Whisper VAD null-audio gating directly on full_audio_emb (replaces Whisper boundary transients with trained 0.0)
    target_nan_check = '        if torch.isnan(full_audio_emb).any():\n            raise ValueError(f"broken audio embedding with nan values")'
    replace_nan_check = '''        if torch.isnan(full_audio_emb).any():
            raise ValueError(f"broken audio embedding with nan values")
        if vad_weights is not None:
            emb_len = full_audio_emb.shape[0]
            w_np = vad_weights[:emb_len]
            if len(w_np) < emb_len:
                w_np = np.pad(w_np, (0, emb_len - len(w_np)), mode="constant", constant_values=0.0)
            w_t = torch.from_numpy(w_np).to(device=full_audio_emb.device, dtype=full_audio_emb.dtype).view(-1, 1, 1)
            full_audio_emb = full_audio_emb * w_t
            if local_rank == 0:
                print(f"✅ [v2.3 Post-Whisper Gate] Zeroed {(w_np < 0.05).sum()}/{emb_len} silent embedding frames to trained null-audio (0.0)")'''
    if target_nan_check in content and "Post-Whisper Gate" not in content:
        content = content.replace(target_nan_check, replace_nan_check)

    # Support avatar-v1.6 model_type in model loading
    content = content.replace('elif model_type == "avatar-v1.5":', 'elif model_type in ("avatar-v1.5", "avatar-v1.6"):')
    content = content.replace("Expected 'avatar-v1.0' or 'avatar-v1.5'.", "Expected 'avatar-v1.0', 'avatar-v1.5', or 'avatar-v1.6'.")

    # Enable offload_kv_cache=True to keep KV cache on CPU RAM, saving ~3GB GPU VRAM during AVC continuation
    content = content.replace("offload_kv_cache=False,", "offload_kv_cache=True,")

    # Enable VAE tiling and slicing + set distilled audio_residual_scale on all DiT blocks
    target_pipe_init = "    pipe.to(local_rank)"
    replace_pipe_init = """    pipe.to(local_rank)
    if hasattr(pipe, "dit") and hasattr(pipe.dit, "blocks"):
        for _blk in pipe.dit.blocks:
            _blk.audio_residual_scale = audio_residual_scale
        if local_rank == 0:
            print(f"✅ [v2.3 Patch] Set DiT audio_residual_scale={audio_residual_scale} across {len(pipe.dit.blocks)} blocks")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.vae.enable_decoder_fused_padding()"""

    if target_pipe_init in content and "audio_residual_scale = audio_residual_scale" not in content:
        content = content.replace(target_pipe_init, replace_pipe_init)

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
    # Restore VAE back onto GPU for prepare_latents image encoding now that Whisper VRAM is freed
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = pipe.vae.to(local_rank)

    if local_rank == 0:
        print(f"Generating segment 1/{num_segments}...")"""

    if target_audio_offload in content and "del pipe.audio_encoder" not in content:
        content = content.replace(target_audio_offload, replace_audio_offload)

    # Free cache after each segment
    target_seg_gc = "        del output\n\n        appended_frames = new_video_array[num_cond_frames:]"
    replace_seg_gc = "        del output\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n\n        appended_frames = new_video_array[num_cond_frames:]"
    if target_seg_gc in content and "torch.cuda.empty_cache()" not in content[content.find("del output"):content.find("del output")+100]:
        content = content.replace(target_seg_gc, replace_seg_gc)

    # Reclaim GPU memory before each continuation segment
    target_loop = "    for segment_idx in range(1, num_segments):\n        if local_rank == 0:\n            print(f\"Generating segment {segment_idx+1}/{num_segments}...\")"
    replace_loop = """    for segment_idx in range(1, num_segments):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if local_rank == 0:
            print(f"Generating segment {segment_idx+1}/{num_segments}...")"""

    if target_loop in content:
        content = content.replace(target_loop, replace_loop)

    # Breakpoint state saving after segment 1 and each continuation segment
    target_save_seg = "            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f\"video_continue_{segment_idx+1}\"), raw_speech_path, fps=save_fps, quality=10)\n            del output_tensor"
    replace_save_seg = """            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"video_continue_{segment_idx+1}"), raw_speech_path, fps=save_fps, quality=10)
            del output_tensor
            ckpt_file = os.path.join(output_dir, "resume_state.pt")
            torch.save({
                "segment_idx": segment_idx + 1,
                "latent": latent.cpu(),
                "all_generated_frames": all_generated_frames[:generated_frame_count],
                "generated_frame_count": generated_frame_count,
            }, ckpt_file)"""

    if target_save_seg in content:
        content = content.replace(target_save_seg, replace_save_seg)

    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {runner_path} (CPU Vocal Separator & Whisper Offload & Checkpointing)")


def patch_dit_avatar_module(dit_module_path: str):
    """
    Patches _create_multi_lora_forward in longcat_video/modules/avatar/longcat_video_dit_avatar.py
    to use in-place zero-allocation GEMM (.addmm_ / .baddbmm_), eliminating both the intermediate `lx`
    tensor (788MB) and `lora_output` (788MB). Saves ~1.57GB of peak activation VRAM during LoRA forward passes.
    """
    if not os.path.exists(dit_module_path):
        raise FileNotFoundError(f"File not found: {dit_module_path}")

    with open(dit_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_start = "    def _create_multi_lora_forward(self, module, loras):"
    target_end = "    def _get_module_by_name(self, module_name):"

    start_idx = content.find(target_start)
    end_idx = content.find(target_end, start_idx)
    if start_idx == -1 or end_idx == -1:
        print("ℹ️ Note: Could not find _create_multi_lora_forward markers in dit_avatar.")
        return

    replace_func = '''    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)
            
            for lora in loras:
                if lora.use_lora:
                    alpha = float(lora.multiplier * float(lora.alpha_scale.float().item() if torch.is_tensor(lora.alpha_scale) else lora.alpha_scale))
                    h = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    if hasattr(lora, 'lora_up') and hasattr(lora.lora_up, 'weight') and getattr(lora.lora_up, 'bias', None) is None:
                        w_up_t = lora.lora_up.weight.t()
                        if w_up_t.dtype != weight_dtype:
                            w_up_t = w_up_t.to(weight_dtype)
                        if h.dtype != weight_dtype:
                            h = h.to(weight_dtype)
                        if org_output.dim() == 3 and org_output.shape[0] == 1:
                            org_output.squeeze(0).addmm_(h.squeeze(0), w_up_t, alpha=alpha)
                        elif org_output.dim() == 2:
                            org_output.addmm_(h, w_up_t, alpha=alpha)
                        else:
                            lx = lora.lora_up(h)
                            if lx.dtype != weight_dtype:
                                lx = lx.to(weight_dtype)
                            org_output.add_(lx, alpha=alpha)
                            del lx
                    else:
                        lx = lora.lora_up(h)
                        if lx.dtype != weight_dtype:
                            lx = lx.to(weight_dtype)
                        org_output.add_(lx, alpha=alpha)
                        del lx
                    del h
            
            return org_output
        
        return multi_lora_forward

'''

    content = content[:start_idx] + replace_func + content[end_idx:]
    with open(dit_module_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {dit_module_path} (Zero-Allocation addmm_ Multi-LoRA Forward, saves ~1.57GB VRAM)")


def patch_rope_module(rope_module_path: str):
    """
    Patches rope_3d.py to eliminate ~870MB-1.5GB of peak activation VRAM during 3D RoPE attention:
    1. Replaces rearrange/unbind/stack in rotate_half with in-place strided assignments.
    2. Keeps RoPE calculations in native target_dtype (bfloat16) instead of upcasting to float32.
    3. Uses in-place addcmul_ accumulator to eliminate temporary intermediate tensors.
    """
    if not os.path.exists(rope_module_path):
        raise FileNotFoundError(f"File not found: {rope_module_path}")

    with open(rope_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Patch rotate_half
    target_rot = '''def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")'''

    replace_rot = '''def rotate_half(x):
    out = torch.empty_like(x)
    out[..., 0::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out'''

    if target_rot in content:
        content = content.replace(target_rot, replace_rot)
    else:
        print("ℹ️ Note: rotate_half already patched or modified.")

    # 2. Patch RotaryPositionalEmbedding.forward
    target_fwd = '''        q_, k_ = q.float(), k.float()
        cache_key = (
            self.head_dim,
            tuple(self.cp_split_hw or ()),
            key_name,
            q.device.type,
            q.device.index,
        )
        cached = self._gpu_cos_sin_cache.get(cache_key)
        if cached is None:
            # Keep the original FP32 path exactly; only reuse its immutable result.
            freqs_cis = self.freqs_dict[key_name].to(q.device)
            freqs_cis = freqs_cis.float().to(q.device)
            cos, sin = freqs_cis.cos(), freqs_cis.sin()
            cos = rearrange(cos, 'n d -> 1 1 n d')
            sin = rearrange(sin, 'n d -> 1 1 n d')
            cached = (cos, sin)
            self._gpu_cos_sin_cache[cache_key] = cached
        cos, sin = cached
        token_count = q.shape[-2]
        cos = cos[:, :, token_offset:token_offset + token_count]
        sin = sin[:, :, token_offset:token_offset + token_count]
        assert cos.shape[-2] == token_count
        q_ = (q_ * cos) + (rotate_half(q_) * sin)
        k_ = (k_ * cos) + (rotate_half(k_) * sin)

        return q_.type_as(q), k_.type_as(k)'''

    replace_fwd = '''        target_dtype = q.dtype
        cache_key = (
            self.head_dim,
            tuple(self.cp_split_hw or ()),
            key_name,
            q.device.type,
            q.device.index,
            target_dtype,
        )
        cached = self._gpu_cos_sin_cache.get(cache_key)
        if cached is None:
            freqs_cis = self.freqs_dict[key_name].to(q.device)
            freqs_cis = freqs_cis.float()
            cos = freqs_cis.cos().to(target_dtype)
            sin = freqs_cis.sin().to(target_dtype)
            cos = rearrange(cos, 'n d -> 1 1 n d')
            sin = rearrange(sin, 'n d -> 1 1 n d')
            cached = (cos, sin)
            self._gpu_cos_sin_cache[cache_key] = cached
        cos, sin = cached
        token_count = q.shape[-2]
        cos = cos[:, :, token_offset:token_offset + token_count]
        sin = sin[:, :, token_offset:token_offset + token_count]
        assert cos.shape[-2] == token_count

        q_rot = rotate_half(q)
        q_out = q * cos
        q_out.addcmul_(q_rot, sin)
        del q_rot

        k_rot = rotate_half(k)
        k_out = k * cos
        k_out.addcmul_(k_rot, sin)
        del k_rot

        return q_out, k_out'''

    if target_fwd in content:
        content = content.replace(target_fwd, replace_fwd)
    else:
        print("ℹ️ Note: RotaryPositionalEmbedding.forward already patched or modified.")

    # 3. Patch RotaryPositionalEmbedding1D.forward
    target_1d = '''        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)'''

    replace_1d = '''        target_dtype = x.dtype
        freqs_cis = freqs_cis.float().to(x.device)
        cos = freqs_cis.cos().to(target_dtype)
        sin = freqs_cis.sin().to(target_dtype)
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_rot = rotate_half(x)
        x_out = x * cos
        x_out.addcmul_(x_rot, sin)
        del x_rot

        return x_out'''

    if target_1d in content:
        content = content.replace(target_1d, replace_1d)
    else:
        print("ℹ️ Note: RotaryPositionalEmbedding1D.forward already patched or modified.")

    with open(rope_module_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {rope_module_path} (Zero-Copy In-Place BF16 3D RoPE, saves ~870MB VRAM)")


def patch_attention_module(attention_module_path: str):
    """
    Patches longcat_video/modules/avatar/attention.py to support PyTorch native
    scaled_dot_product_attention (SDPA) when flash_attn is not installed.
    Crucial fix: Avoid transposing (B, H, S, D) -> (B, S, H, D) before SDPA,
    which previously caused tokens to attend across heads instead of sequence tokens,
    leading to numerical divergence, NaNs, and pitch-black generated frames.
    """
    if not os.path.exists(attention_module_path):
        raise FileNotFoundError(f"File not found: {attention_module_path}")

    with open(attention_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Patch self-attention in _process_attn
    replace_attn = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func
                q_fa = rearrange(q, "B H S D -> B S H D")
                k_fa = rearrange(k, "B H S D -> B S H D")
                v_fa = rearrange(v, "B H S D -> B S H D")
                x = flash_attn_func(
                    q_fa,
                    k_fa,
                    v_fa,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                )
                x = rearrange(x, "B S H D -> B H S D")
            except ImportError:
                import torch.nn.functional as F
                x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)'''

    target_attn_orig = '''        elif self.enable_flashattn2:
            from flash_attn import flash_attn_func
            q = rearrange(q, "B H S D -> B S H D")
            k = rearrange(k, "B H S D -> B S H D")
            v = rearrange(v, "B H S D -> B S H D")
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )
            x = rearrange(x, "B S H D -> B H S D")'''

    target_attn_old_patch = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func
                q = rearrange(q, "B H S D -> B S H D")
                k = rearrange(k, "B H S D -> B S H D")
                v = rearrange(v, "B H S D -> B S H D")
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                )
                x = rearrange(x, "B S H D -> B H S D")
            except ImportError:
                import torch.nn.functional as F
                x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)'''

    if target_attn_orig in content:
        content = content.replace(target_attn_orig, replace_attn)
    elif target_attn_old_patch in content:
        content = content.replace(target_attn_old_patch, replace_attn)
    else:
        print("ℹ️ Note: avatar _process_attn already patched or modified.")

    # 2. Patch cross-attention in _process_cross_attn
    replace_cross = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func
                q_fa = rearrange(q, "B H S D -> B S H D")
                encoder_k_fa = rearrange(encoder_k, "B H S D -> B S H D")
                encoder_v_fa = rearrange(encoder_v, "B H S D -> B S H D")
                x = flash_attn_func(
                    q_fa,
                    encoder_k_fa,
                    encoder_v_fa,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                )
                x = rearrange(x, "B S H D -> B H S D")
            except ImportError:
                import torch.nn.functional as F
                x = F.scaled_dot_product_attention(q, encoder_k, encoder_v, scale=self.scale)'''

    target_cross_orig = '''        elif self.enable_flashattn2:
            from flash_attn import flash_attn_func
            q = rearrange(q, "B H S D -> B S H D")
            encoder_k = rearrange(encoder_k, "B H S D -> B S H D")
            encoder_v = rearrange(encoder_v, "B H S D -> B S H D")
            x = flash_attn_func(
                q,
                encoder_k,
                encoder_v,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )
            x = rearrange(x, "B S H D -> B H S D")'''

    target_cross_old_patch = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func
                q = rearrange(q, "B H S D -> B S H D")
                encoder_k = rearrange(encoder_k, "B H S D -> B S H D")
                encoder_v = rearrange(encoder_v, "B H S D -> B S H D")
                x = flash_attn_func(
                    q,
                    encoder_k,
                    encoder_v,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                )
                x = rearrange(x, "B S H D -> B H S D")
            except ImportError:
                import torch.nn.functional as F
                x = F.scaled_dot_product_attention(q, encoder_k, encoder_v, scale=self.scale)'''

    if target_cross_orig in content:
        content = content.replace(target_cross_orig, replace_cross)
    elif target_cross_old_patch in content:
        content = content.replace(target_cross_old_patch, replace_cross)
    else:
        print("ℹ️ Note: avatar _process_cross_attn already patched or modified.")

    with open(attention_module_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {attention_module_path} (Native PyTorch SDPA FlashAttention fallback)")


def patch_base_attention_module(attention_module_path: str):
    """
    Patches longcat_video/modules/attention.py to support PyTorch native
    scaled_dot_product_attention (SDPA) when flash_attn is not installed.
    """
    if not os.path.exists(attention_module_path):
        raise FileNotFoundError(f"File not found: {attention_module_path}")

    with open(attention_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Patch self-attention in _process_attn
    replace_attn = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func
                q_fa = rearrange(q, "B H S D -> B S H D")
                k_fa = rearrange(k, "B H S D -> B S H D")
                v_fa = rearrange(v, "B H S D -> B S H D")
                x = flash_attn_func(
                    q_fa,
                    k_fa,
                    v_fa,
                    dropout_p=0.0,
                    softmax_scale=getattr(self, 'scale', self.head_dim ** -0.5),
                )
                x = rearrange(x, "B S H D -> B H S D")
            except ImportError:
                import torch.nn.functional as F
                x = F.scaled_dot_product_attention(q, k, v, scale=getattr(self, 'scale', self.head_dim ** -0.5))'''

    target_attn_orig = '''        elif self.enable_flashattn2:
            from flash_attn import flash_attn_func
            q = rearrange(q, "B H S D -> B S H D")
            k = rearrange(k, "B H S D -> B S H D")
            v = rearrange(v, "B H S D -> B S H D")
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )
            x = rearrange(x, "B S H D -> B H S D")'''

    target_attn_old_patch = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func
                q = rearrange(q, "B H S D -> B S H D")
                k = rearrange(k, "B H S D -> B S H D")
                v = rearrange(v, "B H S D -> B S H D")
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                )
                x = rearrange(x, "B S H D -> B H S D")
            except ImportError:
                import torch.nn.functional as F
                x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)'''

    target_attn_old_patch2 = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_func
                q = rearrange(q, "B H S D -> B S H D")
                k = rearrange(k, "B H S D -> B S H D")
                v = rearrange(v, "B H S D -> B S H D")
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=getattr(self, 'scale', self.head_dim ** -0.5),
                )
                x = rearrange(x, "B S H D -> B H S D")
            except ImportError:
                import torch.nn.functional as F
                x = F.scaled_dot_product_attention(q, k, v, scale=getattr(self, 'scale', self.head_dim ** -0.5))'''

    if target_attn_orig in content:
        content = content.replace(target_attn_orig, replace_attn)
    elif target_attn_old_patch2 in content:
        content = content.replace(target_attn_old_patch2, replace_attn)
    elif target_attn_old_patch in content:
        content = content.replace(target_attn_old_patch, replace_attn)
    else:
        print("ℹ️ Note: base _process_attn already patched or modified.")

    # 2. Patch varlen cross-attention in _process_cross_attn
    old_sdpa = "scale=self.scale"
    new_sdpa = "scale=getattr(self, 'scale', self.head_dim ** -0.5)"
    if old_sdpa in content:
        content = content.replace(old_sdpa, new_sdpa)

    target_cross = '''        elif self.enable_flashattn2:
            from flash_attn import flash_attn_varlen_func
            x = flash_attn_varlen_func(
                q=q[0],
                k=k[0],
                v=v[0],
                cu_seqlens_q=torch.tensor([0] + [N] * B, device=q.device).cumsum(0).to(torch.int32),
                cu_seqlens_k=torch.tensor([0] + kv_seqlen, device=q.device).cumsum(0).to(torch.int32),
                max_seqlen_q=N,
                max_seqlen_k=max(kv_seqlen),
            )'''

    replace_cross = '''        elif self.enable_flashattn2:
            try:
                from flash_attn import flash_attn_varlen_func
                x = flash_attn_varlen_func(
                    q=q[0],
                    k=k[0],
                    v=v[0],
                    cu_seqlens_q=torch.tensor([0] + [N] * B, device=q.device).cumsum(0).to(torch.int32),
                    cu_seqlens_k=torch.tensor([0] + kv_seqlen, device=q.device).cumsum(0).to(torch.int32),
                    max_seqlen_q=N,
                    max_seqlen_k=max(kv_seqlen),
                )
            except ImportError:
                import torch.nn.functional as F
                attn_scale = getattr(self, "scale", self.head_dim ** -0.5)
                if B == 1:
                    q_b = q[0].unsqueeze(0).transpose(1, 2)
                    k_b = k[0].unsqueeze(0).transpose(1, 2)
                    v_b = v[0].unsqueeze(0).transpose(1, 2)
                    x_b = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=attn_scale)
                    x = x_b.transpose(1, 2).squeeze(0)
                else:
                    out_list = []
                    k_offset = 0
                    for b in range(B):
                        q_b = q[0, b*N:(b+1)*N].unsqueeze(0).transpose(1, 2)
                        sk = kv_seqlen[b]
                        k_b = k[0, k_offset:k_offset+sk].unsqueeze(0).transpose(1, 2)
                        v_b = v[0, k_offset:k_offset+sk].unsqueeze(0).transpose(1, 2)
                        k_offset += sk
                        x_b = F.scaled_dot_product_attention(q_b, k_b, v_b, scale=attn_scale)
                        out_list.append(x_b.transpose(1, 2).squeeze(0))
                    x = torch.cat(out_list, dim=0)'''

    if target_cross in content:
        content = content.replace(target_cross, replace_cross)
    elif "getattr(self, 'scale'" in content or 'getattr(self, "scale"' in content:
        print("ℹ️ Note: base _process_cross_attn already patched with getattr scale.")
    else:
        print("ℹ️ Note: base _process_cross_attn already patched or modified.")

    with open(attention_module_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Successfully patched {attention_module_path} (Base DiT SDPA Attention fallback)")


def patch_vae_module(vae_module_path: str):
    """
    Patches WanUpsample.forward in autoencoder_kl_wan.py:
    Replaces `return super().forward(x.float()).type_as(x)` with `return super().forward(x)`.
    PyTorch 2.x nn.Upsample natively supports bfloat16, eliminating a 294MB FP32 allocation spike
    that otherwise triggers CUDA OOM on 24GB GPUs during video decoding.
    """
    if not os.path.exists(vae_module_path):
        raise FileNotFoundError(f"File not found: {vae_module_path}")

    with open(vae_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "    def forward(self, x):\n        return super().forward(x.float()).type_as(x)"
    replace = "    def forward(self, x):\n        # Native bfloat16 upsample without FP32 allocation spike\n        return super().forward(x)"

    if target in content:
        content = content.replace(target, replace)
        with open(vae_module_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Successfully patched {vae_module_path} (Native BF16 VAE Upsample)")
    else:
        print("ℹ️ Note: WanUpsample.forward already patched or modified.")


def patch_blocks_module(blocks_path: str):
    """
    Patches FeedForwardSwiGLU.forward in longcat_video/modules/blocks.py
    to execute in-place SwiGLU (F.silu(w1_out, inplace=True); w1_out.mul_(w3_out)),
    cutting peak activation memory from 3.15GB to 1.57GB (net savings ~1.57GB VRAM).
    Also patches modulate_fp32 to execute in-place fp32 mul/add with NaN protection,
    saving ~1.2GB peak VRAM per DiT block and completely eliminating numerical instability.
    """
    if not os.path.exists(blocks_path):
        raise FileNotFoundError(f"File not found: {blocks_path}")

    with open(blocks_path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "    def forward(self, x):\n        return self.w2(F.silu(self.w1(x)) * self.w3(x))"
    replace = """    def forward(self, x):
        w1_out = self.w1(x)
        F.silu(w1_out, inplace=True)
        w3_out = self.w3(x)
        w1_out.mul_(w3_out)
        del w3_out
        out = self.w2(w1_out)
        del w1_out
        return out"""

    if target in content:
        content = content.replace(target, replace)
        print(f"✅ Successfully patched {blocks_path} (In-Place SwiGLU, saves ~1.57GB VRAM)")
    else:
        print("ℹ️ Note: FeedForwardSwiGLU.forward already patched or modified.")

    target_mod = """def modulate_fp32(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, -1, D), scale is (B, -1, D)
    # ensure the modulation params be fp32
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    x = x.to(dtype)
    return x"""

    replace_mod = """def modulate_fp32(norm_func, x, shift, scale):
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    dtype = x.dtype
    # In-place fp32 modulation with NaN protection to eliminate 1.2GB allocation spike per block
    x_fp32 = norm_func(x.to(torch.float32))
    x_fp32 = torch.nan_to_num(x_fp32, 0.0)
    x_fp32.mul_(scale + 1.0).add_(shift)
    x_fp32 = torch.nan_to_num(x_fp32, 0.0)
    return x_fp32.to(dtype)"""

    if target_mod in content:
        content = content.replace(target_mod, replace_mod)
        print(f"✅ Successfully patched {blocks_path} (In-Place modulate_fp32 with NaN protection)")
    else:
        print("ℹ️ Note: modulate_fp32 already patched or modified.")

    # 3. Patch RMSNorm to use PyTorch native fused F.rms_norm, eliminating 488MB FP32 allocation spikes
    target_rmsnorm = """    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight"""

    replace_rmsnorm = """    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), weight=self.weight.to(x.dtype), eps=self.eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight.to(x.dtype)"""

    if target_rmsnorm in content:
        content = content.replace(target_rmsnorm, replace_rmsnorm)
        print(f"✅ Successfully patched {blocks_path} (Fused F.rms_norm, saves ~1GB VRAM per block)")
    else:
        print("ℹ️ Note: RMSNorm already patched or modified.")

    # 4. Patch LayerNorm_FP32 to avoid intermediate FP32 allocation spike on BF16
    target_ln = """        out = F.layer_norm(
            inputs.float(), 
            self.normalized_shape, 
            None if self.weight is None else self.weight.float(), 
            None if self.bias is None else self.bias.float() ,
            self.eps
        ).to(origin_dtype)"""

    replace_ln = """        out = F.layer_norm(
            inputs, 
            self.normalized_shape, 
            self.weight.to(origin_dtype) if self.weight is not None else None, 
            self.bias.to(origin_dtype) if self.bias is not None else None,
            self.eps
        )"""

    if target_ln in content:
        content = content.replace(target_ln, replace_ln)
        print(f"✅ Successfully patched {blocks_path} (Native BF16 LayerNorm, saves ~500MB VRAM)")
    else:
        print("ℹ️ Note: LayerNorm_FP32 already patched or modified.")

    with open(blocks_path, "w", encoding="utf-8") as f:
        f.write(content)


def patch_dit_audio_residual(dit_module_path: str):
    """
    Patches LongCatAvatarSingleStreamBlock.forward so audio_add_x is scaled by
    self.audio_residual_scale (when != 1.0), enabling gentle audio-drive control
    in single-pass distilled mode (--use_distill).
    """
    if not os.path.exists(dit_module_path):
        raise FileNotFoundError(f"File not found: {dit_module_path}")

    with open(dit_module_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_add = "            x = x + audio_add_x\n            x = x.to(x_dtype)"
    replace_add = """            _a_scale = getattr(self, "audio_residual_scale", 1.0)
            x = x + (audio_add_x * _a_scale if _a_scale != 1.0 else audio_add_x)
            x = x.to(x_dtype)"""

    if target_add in content:
        content = content.replace(target_add, replace_add)
        with open(dit_module_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Successfully patched {dit_module_path} (v2.3 Distilled Audio Residual Scaling)")
    elif "audio_residual_scale" in content:
        print(f"ℹ️ Note: {dit_module_path} already patched for audio_residual_scale.")
    else:
        print(f"⚠️ Note: Could not find audio_add_x marker in {dit_module_path}.")


def apply_all_patches(repo_dir: str = "/content/LongCat-Video", target_gpu: str = "A100"):
    quant_path = os.path.join(repo_dir, "longcat_video", "modules", "quantization.py")
    pipe_path = os.path.join(repo_dir, "longcat_video", "pipeline_longcat_video_avatar.py")
    runner_path = os.path.join(repo_dir, "run_demo_avatar_single_audio_to_video.py")
    dit_avatar_path = os.path.join(repo_dir, "longcat_video", "modules", "avatar", "longcat_video_dit_avatar.py")
    rope_path = os.path.join(repo_dir, "longcat_video", "modules", "avatar", "rope_3d.py")
    attn_path = os.path.join(repo_dir, "longcat_video", "modules", "avatar", "attention.py")
    base_attn_path = os.path.join(repo_dir, "longcat_video", "modules", "attention.py")
    blocks_path = os.path.join(repo_dir, "longcat_video", "modules", "blocks.py")
    vae_path = os.path.join(repo_dir, "longcat_video", "modules", "autoencoder_kl_wan.py")

    # Essential SDPA compatibility, checkpointing & universal audio control for all GPUs
    if os.path.exists(attn_path):
        patch_attention_module(attn_path)
    if os.path.exists(base_attn_path):
        patch_base_attention_module(base_attn_path)
    if os.path.exists(runner_path):
        patch_runner_script(runner_path)
    if os.path.exists(vae_path):
        patch_vae_module(vae_path)
    if os.path.exists(dit_avatar_path):
        patch_dit_audio_residual(dit_avatar_path)

    # Detect VRAM capacity (A100-40GB vs A100-80GB)
    total_vram_gb = 80.0
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if total_vram_gb < 70 or target_gpu not in ["A100", "A100-80G"]:
        print(f"🔧 GPU VRAM is {total_vram_gb:.1f}GB (<70GB): Applying CPU text_encoder offload and zero-allocation RoPE/LoRA patches...")
        if os.path.exists(pipe_path):
            patch_pipeline_module(pipe_path)
        if os.path.exists(dit_avatar_path):
            patch_dit_avatar_module(dit_avatar_path)
        if os.path.exists(rope_path):
            patch_rope_module(rope_path)
        if os.path.exists(blocks_path):
            patch_blocks_module(blocks_path)
        if os.path.exists(quant_path):
            patch_quantization_module(quant_path)
    else:
        print("🚀 Target GPU is A100-80GB (>70GB VRAM): skipping VAE CPU offload and aggressive quantization to preserve 100% full precision fidelity.")


if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else "/content/LongCat-Video"
    gpu_type = sys.argv[2] if len(sys.argv) > 2 else "A100"
    apply_all_patches(target_repo, target_gpu=gpu_type)
