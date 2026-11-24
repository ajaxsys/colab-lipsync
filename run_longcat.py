"""
Autonomous Runner for LongCat-Video-Avatar 1.6 on 24GB GPUs (NVIDIA L4 / RTX 4090).

Pipeline features:
1. Auto-applies zero-RAM meta-device patch to avoid CPU memory OOM (SIGKILL).
2. Auto-applies text-encoder CPU offload patch to fit within 24GB VRAM without OOM.
3. Auto-applies vocal separator CPU patch to avoid ONNX Runtime CUDA BFC arena OOM.
4. Auto-applies Whisper audio-encoder CPU offload to free 3.1GB VRAM for DiT forward pass.
5. Uses distilled 8-step inference with scale=1.0 (CFG=False) to halve activation memory.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None, env=None):
    print(f"🚀 Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    # Use start_new_session=True to detach child process group and prevent SIGHUP on SSH disconnect
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), env=run_env, start_new_session=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with return code {res.returncode}")


def main():
    base_dir = Path(__file__).resolve().parent
    default_image = str(base_dir / "sample" / "girl_9x16.jpeg")
    default_audio = str(base_dir / "sample" / "audio_cut_30s.mp3")

    parser = argparse.ArgumentParser(description="Run LongCat-Video-Avatar 1.6 with zero-RAM and CPU-offload patches")
    parser.add_argument("--image", default=default_image, help="Input portrait image path")
    parser.add_argument("--audio", default=default_audio, help="Input audio singing file path (master full track)")
    parser.add_argument("--vocal_audio", default="", help="Input pre-separated pure vocal audio path (optional, overrides internal separator)")
    parser.add_argument("--output_dir", default="./output_longcat", help="Output directory")
    parser.add_argument("--num_segments", type=int, default=10, help="Number of segments (10 for ~30s, 1 for rapid smoke test)")
    parser.add_argument("--resolution", default="480p", choices=["480p", "720p"], help="Video resolution")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Distilled inference steps (default: 8)")
    parser.add_argument("--text_guidance_scale", type=float, default=1.0, help="Text guidance scale (default: 1.0 for distilled)")
    parser.add_argument("--audio_guidance_scale", type=float, default=0.42, help="Audio guidance scale (tuned down to 0.42 for gentle, natural lip movements)")
    parser.add_argument("--mix_steps", type=int, default=0, help="Distill LoRA mixing steps (0 to bypass DMD and solely use smooth OPSD loraB)")
    parser.add_argument("--mix_alpha", type=float, default=0.0, help="Distill LoRA mixing alpha (0.0 for pure OPSD)")
    parser.add_argument("--use_int8", action="store_true", help="Enable INT8 quantization (only needed on 24GB GPUs; disable on A100/H100)")
    parser.add_argument("--target_gpu", default="A100", choices=["A100", "L4", "4090"], help="Target GPU architecture (default: A100)")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    audio_path = os.path.abspath(args.audio)
    drive_audio = os.path.abspath(args.vocal_audio) if args.vocal_audio else audio_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if args.vocal_audio and not os.path.exists(drive_audio):
        raise FileNotFoundError(f"Vocal audio not found: {drive_audio}")

    repo_dir = "/content/LongCat-Video"
    weights_dir = f"{repo_dir}/weights/LongCat-Video-Avatar-1.6"
    job_json_path = "/content/job_longcat.json"

    # 1. Create job json using drive_audio (pure vocal) for DiT lipsync conditioning
    job_data = {
        "prompt": "A beautiful girl singing gently and softly, elegant subtle mouth movements, lyrical expression, ultra realistic skin and hair, cinematic stage lighting, 8k masterpiece",
        "cond_image": image_path,
        "cond_audio": {
            "person1": drive_audio
        }
    }
    with open(job_json_path, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)
    print(f"✅ Generated job config: {job_json_path} (drive audio: {drive_audio})")

    # 2. Apply memory and VRAM patches
    patch_module_path = Path(__file__).parent / "patches" / "patch_longcat.py"
    if patch_module_path.exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from patches.patch_longcat import apply_all_patches
        apply_all_patches(repo_dir, target_gpu=args.target_gpu)

    # 3. Launch torchrun with PyTorch memory fragmentation configuration
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }

    cmd = [
        "torchrun", "--nproc_per_node=1", "run_demo_avatar_single_audio_to_video.py",
        f"--checkpoint_dir={weights_dir}",
        "--stage_1=ai2v",
        f"--input_json={job_json_path}",
        f"--output_dir={args.output_dir}",
        "--model_type=avatar-v1.6",
        "--use_distill",
        "--lora_path=loraB/lora.safetensors",
        f"--mix_steps={args.mix_steps}",
        f"--mix_alpha={args.mix_alpha}",
        f"--resolution={args.resolution}",
        f"--num_segments={args.num_segments}",
        f"--num_inference_steps={args.num_inference_steps}",
        f"--text_guidance_scale={args.text_guidance_scale}",
        f"--audio_guidance_scale={args.audio_guidance_scale}",
        "--save_every_segment"
    ]
    if args.use_int8:
        cmd.append("--use_int8")
    run_cmd(cmd, cwd=repo_dir, env=env_vars)


if __name__ == "__main__":
    main()
