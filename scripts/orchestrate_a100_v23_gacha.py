#!/usr/bin/env python3
"""
Orchestrator for LongCat Avatar v2.3 Multi-Seed Statistical Gacha Evaluation on Colab A100.
Runs 4 independent random seeds (42, 123, 777, 2026) on Segment 1+2 (~6.92s each) using
strictly universal (zero hardcoded timestamp) v2.3 logic:
  - Frame-level VAD with post-Whisper null-audio gating (0.0 on silent frames)
  - Neutral non-mouth-inducing text prompt
  - Distilled DiT audio residual scaling (0.55)
  - Pure OPSD (mix_steps=4, mix_alpha=0.0)
"""

import os
import sys
import time
import subprocess

SESSION = "a100-v23-gacha"
SEEDS = [42, 123, 777, 2026]
AUDIO_SCALE = 0.55


def run(cmd, check=True, capture=True):
    print(f"\n[EXEC] {cmd[:160]}...")
    res = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and res.returncode != 0:
        err = res.stderr.strip() if capture else "Command failed"
        print(f"❌ Error: {err[:300]}")
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd[:100]}")
    return res


def ensure_session():
    res = run("colab sessions", capture=True)
    if SESSION in res.stdout:
        print(f"✅ Session '{SESSION}' already exists and is active!")
        return True

    print(f"🔄 Requesting A100 session '{SESSION}'...")
    max_retries = 30
    for i in range(1, max_retries + 1):
        flag = "--high-mem" if (i % 2 == 0) else ""
        print(f"[{i}/{max_retries}] Attempting 'colab new --gpu A100 {flag} -s {SESSION}'...")
        res = subprocess.run(
            f"colab new --gpu A100 {flag} -s {SESSION}",
            shell=True,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print(f"🎉 Successfully created A100 session on attempt {i}!")
            return True
        err = res.stderr.strip().replace("\n", " ")
        print(f"   Waiting 25s for A100 pool availability (error: {err[:80]}...)")
        time.sleep(25)
    raise RuntimeError("Failed to provision A100 after maximum retries.")


def setup_remote():
    print("\n📦 Setting up remote A100 environment...")
    remote_setup_sh = r"""
set -e
export DEBIAN_FRONTEND=noninteractive

echo '=== 1. PACKAGES ==='
apt-get update -qq
apt-get install -y -qq tmux ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev git git-lfs aria2 >/dev/null

pip install -q av==19.0.2 diffusers==0.35.1 huggingface_hub==0.36.2 torchvision==0.29.2 onnxruntime-gpu==1.32.0 audio-separator soundfile einops accelerate transformers==4.49.0 loguru ftfy imageio imageio-ffmpeg librosa pyloudnorm nvidia-ml-py soxr opencv-python-headless

echo '=== 2. FIX NCCL ==='
pip install -q --force-reinstall nvidia-nccl-cu13 || true
NCCL_SO=$(find /usr/local/lib/ -name 'libnccl.so.2*' 2>/dev/null | head -n 1)
if [ -n "$NCCL_SO" ]; then
    cp -P $(dirname "$NCCL_SO")/libnccl.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
    ldconfig
fi

echo '=== 3. REPOSITORIES ==='
rm -rf /content/colab-lipsync /content/LongCat-Video
git clone -b feat/pipeline-upgrade-v2 https://github.com/ajaxsys/colab-lipsync.git /content/colab-lipsync
git clone https://github.com/meituan-longcat/LongCat-Video.git /content/LongCat-Video

echo '=== 4. VERIFY GPU ==='
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0), 'VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')"
echo '=== SETUP COMPLETED ==='
"""
    with open("/tmp/remote_setup_v23.sh", "w") as f:
        f.write(remote_setup_sh)
    run(f"colab upload -s {SESSION} /tmp/remote_setup_v23.sh /content/remote_setup.sh")
    run(f"colab ssh -s {SESSION} -- 'bash /content/remote_setup.sh'", capture=False)


def download_weights():
    print("\n📥 Downloading weights on A100...")
    weight_py = r"""
import os, time
from huggingface_hub import snapshot_download

start = time.time()
print('🚀 [1/2] Downloading base components...')
snapshot_download(
    repo_id='meituan-longcat/LongCat-Video',
    local_dir='/content/LongCat-Video/weights/LongCat-Video',
    allow_patterns=['text_encoder/*', 'tokenizer/*', 'vae/*', 'model_index.json'],
    max_workers=8
)

print('🚀 [2/2] Downloading Avatar 1.6 BF16 weights...')
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
    print('🔗 Linked loraB to loraA (for pure OPSD dynamic LoRA dispatch)')

elapsed = round(time.time() - start, 1)
print(f'🎉 Weights downloaded in {elapsed}s')
"""
    with open("/tmp/download_weights_v23.py", "w") as f:
        f.write(weight_py)
    run(f"colab upload -s {SESSION} /tmp/download_weights_v23.py /content/download_weights.py")
    run(f"colab ssh -s {SESSION} -- 'python3 /content/download_weights.py'", capture=False)


def start_gacha_batch():
    print(f"\n🎲 Launching {len(SEEDS)}-seed gacha batch inside detached tmux...")
    seed_list_str = " ".join(str(s) for s in SEEDS)
    launch_sh = f"""
tmux kill-session -t gacha 2>/dev/null || true
cat << 'INNER' > /content/run_gacha_batch.sh
#!/bin/bash
set -e
for SEED in {seed_list_str}; do
    echo "=========================================="
    echo "🎲 [GACHA] Running SEED=$SEED (audio_scale={AUDIO_SCALE})"
    echo "=========================================="
    OUT_DIR="/content/output_v23_seed${{SEED}}"
    mkdir -p "$OUT_DIR"
    cd /content/LongCat-Video
    git checkout . >/dev/null 2>&1 || true
    python3 -u /content/colab-lipsync/run_longcat.py \\
        --target_gpu=A100 \\
        --resolution=480p \\
        --num_segments=2 \\
        --audio_guidance_scale={AUDIO_SCALE} \\
        --mix_steps=4 \\
        --mix_alpha=0.0 \\
        --seed="$SEED" \\
        --output_dir="$OUT_DIR"
done
echo "🎉 ALL_GACHA_DRAWS_COMPLETED"
INNER
chmod +x /content/run_gacha_batch.sh
tmux new-session -d -s gacha "bash /content/run_gacha_batch.sh 2>&1 | tee /content/gacha_batch.log"
sleep 3
tmux list-sessions
echo '--- GACHA BATCH LAUNCHED IN TMUX ---'
"""
    with open("/tmp/launch_gacha.sh", "w") as f:
        f.write(launch_sh)
    run(f"colab upload -s {SESSION} /tmp/launch_gacha.sh /content/launch_gacha.sh")
    run(f"colab ssh -s {SESSION} -- 'bash /content/launch_gacha.sh'", capture=False)


def monitor_and_download():
    print("\n👀 Monitoring 4-seed gacha batch until completion...")
    start_time = time.time()
    last_log = ""
    while True:
        time.sleep(45)
        tmux_res = run(f"colab ssh -s {SESSION} -- 'tmux has-session -t gacha 2>/dev/null'", check=False)
        is_running = (tmux_res.returncode == 0)

        log_res = run(
            f"colab ssh -s {SESSION} -- \"tail -n 30 /content/gacha_batch.log | tr '\\r' '\\n' | tail -n 18\"",
            check=False,
        )
        curr_log = log_res.stdout.strip()
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed//60}m {elapsed%60}s] TMUX running: {is_running}")
        if curr_log and curr_log != last_log:
            print(f"--- LOG ---\n{curr_log}\n-----------")
            last_log = curr_log

        if not is_running:
            break

    # Verify & download each generated video
    downloaded = []
    for seed in SEEDS:
        remote_mp4 = f"/content/output_v23_seed{seed}/video_continue_2.mp4"
        chk = run(f"colab ssh -s {SESSION} -- \"ls -lh {remote_mp4} 2>/dev/null\"", check=False)
        if "video_continue_2.mp4" in chk.stdout:
            local_mp4 = os.path.abspath(f"sample/v23_gacha_seed{seed}.mp4")
            os.makedirs(os.path.dirname(local_mp4), exist_ok=True)
            run(f"colab download -s {SESSION} {remote_mp4} {local_mp4}", capture=False)
            print(f"✅ Downloaded seed {seed} -> {local_mp4} ({os.path.getsize(local_mp4)/1024**2:.1f} MB)")
            downloaded.append(local_mp4)
        else:
            print(f"⚠️ Missing output for seed {seed}!")

    if not downloaded:
        dump_res = run(f"colab ssh -s {SESSION} -- \"tail -n 120 /content/gacha_batch.log\"", check=False)
        print(dump_res.stdout)
        raise RuntimeError("No gacha videos were produced.")

    # Save remote log excerpt for inspection
    run(f"colab download -s {SESSION} /content/gacha_batch.log sample/v23_gacha_batch.log", check=False)


def teardown():
    print(f"\n🛑 Stopping Colab session '{SESSION}' to prevent billing...")
    run(f"colab stop -s {SESSION}", check=False, capture=False)
    print("✅ Compute units frozen. Teardown complete.")


def main():
    try:
        ensure_session()
        setup_remote()
        download_weights()
        start_gacha_batch()
        monitor_and_download()
    finally:
        teardown()


if __name__ == "__main__":
    main()
