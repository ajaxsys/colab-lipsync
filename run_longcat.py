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
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), env=run_env)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with return code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run LongCat-Video-Avatar 1.6 with zero-RAM and CPU-offload patches")
    parser.add_argument("--image", required=True, help="Input portrait image path")
    parser.add_argument("--audio", required=True, help="Input audio singing file path")
    parser.add_argument("--output_dir", default="./output_longcat", help="Output directory")
    parser.add_argument("--num_segments", type=int, default=10, help="Number of segments (10 for ~30s)")
    parser.add_argument("--resolution", default="480p", choices=["480p", "720p"], help="Video resolution")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Distilled inference steps (default: 8)")
    parser.add_argument("--text_guidance_scale", type=float, default=1.0, help="Text guidance scale (default: 1.0 for distilled)")
    parser.add_argument("--audio_guidance_scale", type=float, default=1.0, help="Audio guidance scale (default: 1.0 for distilled)")
    args = parser.parse_args()

    repo_dir = "/content/LongCat-Video"
    weights_dir = f"{repo_dir}/weights/LongCat-Video-Avatar-1.6"
    job_json_path = "/content/job_longcat.json"

    # 1. Create job json
    job_data = {
        "prompt": "A beautiful girl singing emotionally, ultra realistic, detailed facial expression, cinematic lighting, 8k masterpiece",
        "cond_image": os.path.abspath(args.image),
        "cond_audio": {
            "person1": os.path.abspath(args.audio)
        }
    }
    with open(job_json_path, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)
    print(f"✅ Generated job config: {job_json_path}")

    # 2. Apply memory and VRAM patches
    patch_module_path = Path(__file__).parent / "patches" / "patch_longcat.py"
    if patch_module_path.exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from patches.patch_longcat import apply_all_patches
        apply_all_patches(repo_dir)

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
        "--mix_steps=4",
        "--mix_alpha=1.0",
        "--use_int8",
        f"--resolution={args.resolution}",
        f"--num_segments={args.num_segments}",
        f"--num_inference_steps={args.num_inference_steps}",
        f"--text_guidance_scale={args.text_guidance_scale}",
        f"--audio_guidance_scale={args.audio_guidance_scale}"
    ]
    run_cmd(cmd, cwd=repo_dir, env=env_vars)


if __name__ == "__main__":
    main()
