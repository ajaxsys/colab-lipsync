#!/usr/bin/env python3
"""
Orchestrator for LongCat Avatar v2.2 Full 30s Render on Colab A100.
Handles:
1. Resilient A100 session provisioning with retry backoff.
2. Fast environment setup & weight downloading (BF16 native, loraB symlink).
3. 10-segment (30.48s) inference execution with v2.2 parameters (pure OPSD, scale 0.42, vocal limiter).
4. Continuous monitoring & progress reporting.
5. Result download to sample/output_longcat_avatar_30s_v22.mp4.
6. Immediate shutdown of A100 to freeze compute unit costs.
"""

import os
import sys
import time
import subprocess

SESSION = "a100-v22-full"
OUTPUT_LOCAL = os.path.abspath("sample/output_longcat_avatar_30s_v22.mp4")

def run(cmd, check=True, capture=True):
    print(f"\n[EXEC] {cmd[:160]}...")
    res = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and res.returncode != 0:
        err = res.stderr.strip() if capture else "Command failed"
        print(f"❌ Error: {err[:300]}")
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd[:100]}")
    return res

def ensure_session():
    # Check if session already exists
    res = run("colab sessions", capture=True)
    if SESSION in res.stdout:
        print(f"✅ Session '{SESSION}' already exists and is active!")
        return True

    print(f"🔄 Requesting A100 session '{SESSION}'...")
    max_retries = 30
    for i in range(1, max_retries + 1):
        flag = "--high-mem" if (i % 2 == 0) else ""
        print(f"[{i}/{max_retries}] Attempting 'colab new --gpu A100 {flag} -s {SESSION}'...")
        res = subprocess.run(f"colab new --gpu A100 {flag} -s {SESSION}", shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"🎉 Successfully created A100 session on attempt {i}!")
            return True
        else:
            err = res.stderr.strip().replace("\n", " ")
            print(f"   Waiting 25s for A100 pool availability (error: {err[:80]}...)")
            time.sleep(25)
    raise RuntimeError("Failed to provision A100 after maximum retries.")

def setup_remote():
    print("\n📦 Setting up remote A100 environment...")
    setup_cmd = f"""colab ssh -s {SESSION} -- "
set -e
export DEBIAN_FRONTEND=noninteractive

echo '=== 1. PACKAGES ==='
apt-get update -qq
apt-get install -y -qq ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev git git-lfs aria2 >/dev/null

pip install -q av==19.0.2 diffusers==0.35.1 huggingface_hub==0.36.2 torchvision==0.29.2 onnxruntime-gpu==1.32.0 audio-separator soundfile einops accelerate transformers==4.49.0 loguru ftfy imageio imageio-ffmpeg librosa pyloudnorm nvidia-ml-py soxr opencv-python-headless

echo '=== 2. FIX NCCL ==='
pip install -q --force-reinstall nvidia-nccl-cu13 || true
NCCL_SO=\\$(find /usr/local/lib/ -name 'libnccl.so.2*' 2>/dev/null | head -n 1)
if [ -n \\\"\\$NCCL_SO\\\" ]; then
    cp -P \\$(dirname \\\"\\$NCCL_SO\\\")/libnccl.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    ldconfig
fi

echo '=== 3. REPOSITORIES ==='
rm -rf /content/colab-lipsync /content/LongCat-Video
git clone -b feat/pipeline-upgrade-v2 https://github.com/ajaxsys/colab-lipsync.git /content/colab-lipsync
git clone https://github.com/meituan-longcat/LongCat-Video.git /content/LongCat-Video

echo '=== 4. VERIFY GPU ==='
python3 -c '
import torch
print(f\\\"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB\\\")
'
" """
    run(setup_cmd, capture=False)

def download_weights():
    print("\n📥 Downloading weights on A100...")
    weight_script = f"""colab ssh -s {SESSION} -- "
cat << 'EOF' > /content/download_a100_weights.py
import os, time
from huggingface_hub import snapshot_download

start = time.time()
print('🚀 Downloading base components...')
snapshot_download(
    repo_id='meituan-longcat/LongCat-Video',
    local_dir='/content/LongCat-Video/weights/LongCat-Video',
    allow_patterns=['text_encoder/*', 'tokenizer/*', 'vae/*', 'model_index.json'],
    max_workers=8
)

print('🚀 Downloading Avatar 1.6 BF16 weights...')
snapshot_download(
    repo_id='meituan-longcat/LongCat-Video-Avatar-1.6',
    local_dir='/content/LongCat-Video/weights/LongCat-Video-Avatar-1.6',
    allow_patterns=[
        'base_model/*',
        'whisper-large-v3/*',
        'vocal_separator/*',
        'scheduler/*',
        'loraB/*'
    ],
    max_workers=8
)

os.makedirs('/content/LongCat-Video/weights/LongCat-Video-Avatar-1.6/loraA', exist_ok=True)
lora_a = '/content/LongCat-Video/weights/LongCat-Video-Avatar-1.6/loraA/lora.safetensors'
lora_b = '/content/LongCat-Video/weights/LongCat-Video-Avatar-1.6/loraB/lora.safetensors'
if os.path.exists(lora_b):
    if os.path.islink(lora_a) or os.path.exists(lora_a):
        os.remove(lora_a)
    os.symlink(lora_b, lora_a)
    print('🔗 Linked loraB to loraA (for dynamic pure OPSD dispatch)')

print(f'🎉 Weights downloaded in {{time.time() - start:.1f}}s')
EOF
python3 /content/download_a100_weights.py
" """
    run(weight_script, capture=False)

def start_inference():
    print("\n🎬 Launching full 30.48s (10 segments) inference...")
    launch_cmd = f"""colab ssh -s {SESSION} -- "
mkdir -p /content/output_longcat_30s_v22
nohup python3 /content/colab-lipsync/run_longcat.py \\
  --target_gpu=A100 \\
  --resolution=480p \\
  --num_segments=10 \\
  --audio_guidance_scale=0.42 \\
  --mix_steps=4 \\
  --mix_alpha=0.0 \\
  --output_dir=/content/output_longcat_30s_v22 > /content/run_longcat_30s.log 2>&1 &
sleep 5
ps aux | grep -E 'run_longcat|torchrun' | grep -v grep
echo '--- INFERENCE LAUNCHED ---'
" """
    run(launch_cmd, capture=False)

def monitor_and_download():
    print("\n👀 Monitoring inference until completion...")
    start_time = time.time()
    last_log = ""
    while True:
        time.sleep(45)
        # Check process status
        ps_res = run(f"colab ssh -s {SESSION} -- \"ps aux | grep -E 'run_longcat|torchrun' | grep -v grep\"", check=False)
        is_running = (ps_res.returncode == 0 and len(ps_res.stdout.strip()) > 0)

        # Check log tail
        log_res = run(f"colab ssh -s {SESSION} -- \"tail -n 25 /content/run_longcat_30s.log | tr '\\r' '\\n' | tail -n 15\"", check=False)
        curr_log = log_res.stdout.strip()
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed//60}m {elapsed%60}s] Running: {is_running}")
        if curr_log and curr_log != last_log:
            print(f"--- LOG ---\n{curr_log}\n-----------")
            last_log = curr_log

        # Check if final video exists
        check_video = run(f"colab ssh -s {SESSION} -- \"ls -lh /content/output_longcat_30s_v22/video_continue_10.mp4 2>/dev/null\"", check=False)
        if "video_continue_10.mp4" in check_video.stdout:
            print("🎉 Found final output video_continue_10.mp4!")
            break

        if not is_running:
            # If not running and video not found, check if it failed or finished under different name
            full_check = run(f"colab ssh -s {SESSION} -- \"ls -lh /content/output_longcat_30s_v22/\"", check=False)
            print(f"Files in output dir:\n{full_check.stdout}")
            if "video_continue" in full_check.stdout:
                print("Inference finished!")
                break
            else:
                print("⚠️ Process died unexpectedly. Printing last 50 lines of log:")
                tail_res = run(f"colab ssh -s {SESSION} -- \"tail -n 50 /content/run_longcat_30s.log\"", check=False)
                print(tail_res.stdout)
                raise RuntimeError("Inference terminated without producing final video.")

    print("\n📥 Downloading final video to local workspace...")
    os.makedirs(os.path.dirname(OUTPUT_LOCAL), exist_ok=True)
    run(f"colab download -s {SESSION} /content/output_longcat_30s_v22/video_continue_10.mp4 {OUTPUT_LOCAL}", capture=False)
    print(f"✅ Downloaded to: {OUTPUT_LOCAL} ({os.path.getsize(OUTPUT_LOCAL)/1024**2:.1f} MB)")

def teardown():
    print(f"\n🛑 Stopping Colab session '{SESSION}' to prevent billing...")
    run(f"colab stop -s {SESSION}", check=False, capture=False)
    print("✅ Compute units frozen. Teardown complete.")

def main():
    try:
        ensure_session()
        setup_remote()
        download_weights()
        start_inference()
        monitor_and_download()
    finally:
        teardown()

if __name__ == "__main__":
    main()
