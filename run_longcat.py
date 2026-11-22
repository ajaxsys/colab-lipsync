"""
Autonomous Runner for LongCat-Video-Avatar 1.6 on 24GB GPUs (NVIDIA L4 / RTX 4090).

Pipeline features:
1. Auto-applies zero-RAM meta-device patch to avoid CPU memory OOM (SIGKILL).
2. Uses INT8 quantized base model and mixed LoRA distillation (8-step inference).
3. Connects Whisper-Large-v3 and Kim_Vocal_2 for noise-immune lip synchronization.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None):
    print(f"🚀 Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with return code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run LongCat-Video-Avatar 1.6 with zero-RAM patch")
    parser.add_argument("--image", required=True, help="Input portrait image path")
    parser.add_argument("--audio", required=True, help="Input audio singing file path")
    parser.add_argument("--output_dir", default="./output_longcat", help="Output directory")
    parser.add_argument("--num_segments", type=int, default=10, help="Number of segments (10 for ~30s)")
    parser.add_argument("--resolution", default="480p", choices=["480p", "720p"], help="Video resolution")
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

    # 2. Apply memory patch
    patch_script = Path(__file__).parent / "patches" / "patch_longcat.py"
    target_py = f"{repo_dir}/longcat_video/modules/quantization.py"
    if os.path.exists(target_py):
        from patches.patch_longcat import patch_quantization_module
        patch_quantization_module(target_py)
    else:
        print(f"⚠️ {target_py} not found, skipping local patch.")

    # 3. Launch torchrun
    cmd = [
        "torchrun", "--nproc_per_node=1", "run_demo_avatar_single_audio_to_video.py",
        f"--checkpoint_dir={weights_dir}",
        "--stage_1=ai2v",
        f"--input_json={job_json_path}",
        f"--output_dir={args.output_dir}",
        "--model_type=avatar-v1.6",
        "--use_distill",
        "--use_int8",
        f"--resolution={args.resolution}",
        f"--num_segments={args.num_segments}"
    ]
    run_cmd(cmd, cwd=repo_dir)


if __name__ == "__main__":
    main()
