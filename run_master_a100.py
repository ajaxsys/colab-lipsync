import os
import sys
import subprocess
import json
import time

def log(msg):
    t = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open("/content/master_a100.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run(cmd, desc=""):
    log(f"=== [START] {desc} ===")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        log(f"=== [FAILED] {desc} (code {res.returncode}) ===")
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    log(f"=== [DONE] {desc} ===")

def main():
    log("🚀 Master pipeline started on Colab A100...")

    # 自动探测并挂载 Google Drive，启用断点接续引擎（防崩溃/防实例回收）
    if os.path.exists("/content") and not os.path.exists("/content/drive/MyDrive"):
        try:
            log("💾 正在检测并挂载 Google Drive 作为持久化断点存储...")
            from google.colab import drive
            drive.mount("/content/drive")
            log("✅ Google Drive 挂载成功！断点续跑已激活。")
        except Exception as e:
            log(f"⚠️ Google Drive 挂载跳过（如需跨实例断点续跑请在页面挂载 Drive）：{e}")

    # 1. Clone repos if needed
    if not os.path.exists("/content/InfiniteTalk"):
        run("git clone https://github.com/MeiGen-AI/InfiniteTalk.git /content/InfiniteTalk", "Clone InfiniteTalk")

    if not os.path.exists("/content/colab-lipsync"):
        run("git clone https://github.com/ajaxsys/colab-lipsync.git /content/colab-lipsync", "Clone colab-lipsync")
    else:
        run("cd /content/colab-lipsync && git pull", "Git pull colab-lipsync")

    # 2. Install requirements
    run("pip install -q -r /content/colab-lipsync/requirements.txt", "Install requirements")

    # 3. Download models (will download models if not present)
    run("cd /content/colab-lipsync && python3 download_models.py", "Download models")

    # Base models dict
    models_dict = {
        "base": "/content/models/base",
        "dit": "/content/models/dit/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        "infinitetalk": "/content/models/infinitetalk/infinitetalk_single.safetensors",
        "lora": "/content/models/lora/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "wav2vec": "/content/models/wav2vec",
        "t5_dir": "/content/models/t5",
        "demucs": "/content/models/demucs/htdemucs.pth",
        "bs_roformer": "/content/models/bs_roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "codeformer": "/content/models/codeformer/codeformer.pth"
    }

    # ========================================================
    # VERSION 1: 480 × 848 (9:16 竖屏)
    # 输入: /content/girl_9x16.jpeg
    # ========================================================
    os.makedirs("/content/workspace_portrait", exist_ok=True)
    out1 = "/content/output_480p_portrait_9_16.mp4"
    job1 = {
        "work_dir": "/content/workspace_portrait",
        "code_dir": "/content/InfiniteTalk",
        "image": "/content/girl_9x16.jpeg",
        "audio": "/content/song.mp3",
        "prompt": "女人正在唱歌，面部生动，眼神自然流转",
        "negative": "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        "out": out1,
        "res": "custom",
        "width": 480,
        "height": 848,
        "steps": 4,
        "seed": 42,
        "start_seconds": 0.0,
        "max_seconds": 0.0,
        "accel": True,
        "sage": True,
        "color_fix": 1.0,
        "separator": "bs_roformer",
        "enhance_face": False,
        "interpolate_fps": 0,
        "checkpoint_dir": "/content/drive/MyDrive/InfiniteTalk_Checkpoints/job_480x848" if os.path.exists("/content/drive/MyDrive") else "/content/workspace_portrait/checkpoints",
        "models": models_dict
    }

    job1_path = "/content/workspace_portrait/job_portrait.json"
    with open(job1_path, "w", encoding="utf-8") as f:
        json.dump(job1, f, ensure_ascii=False, indent=2)

    log("🎬 Starting Version 1: 480 × 848 (9:16 竖屏)...")
    run(f"cd /content/colab-lipsync && python3 -u extracted_runner.py {job1_path} >> /content/generation_portrait.log 2>&1", "Execute Version 1 (480x848)")

    if os.path.exists(out1) and os.path.getsize(out1) > 1024 * 1024:
        log("✅ Version 1 complete! Video generated successfully.")
        with open("/content/STATUS_V1_DONE", "w") as f:
            f.write(f"DONE:{out1}\n")
    else:
        log("❌ Version 1 output file missing or invalid.")
        with open("/content/STATUS_V1_FAILED", "w") as f:
            f.write("FAILED\n")
        return

    # ========================================================
    # VERSION 2: 848 × 480 (16:9 横屏)
    # 输入: /content/girl_16x9.webp
    # ========================================================
    os.makedirs("/content/workspace_landscape", exist_ok=True)
    out2 = "/content/output_480p_landscape_16_9.mp4"
    job2 = {
        "work_dir": "/content/workspace_landscape",
        "code_dir": "/content/InfiniteTalk",
        "image": "/content/girl_16x9.webp",
        "audio": "/content/song.mp3",
        "prompt": "女人正在唱歌，面部生动，眼神自然流转",
        "negative": "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        "out": out2,
        "res": "custom",
        "width": 848,
        "height": 480,
        "steps": 4,
        "seed": 42,
        "start_seconds": 0.0,
        "max_seconds": 0.0,
        "accel": True,
        "sage": True,
        "color_fix": 1.0,
        "separator": "bs_roformer",
        "enhance_face": False,
        "interpolate_fps": 0,
        "checkpoint_dir": "/content/drive/MyDrive/InfiniteTalk_Checkpoints/job_848x480" if os.path.exists("/content/drive/MyDrive") else "/content/workspace_landscape/checkpoints",
        "models": models_dict
    }

    job2_path = "/content/workspace_landscape/job_landscape.json"
    with open(job2_path, "w", encoding="utf-8") as f:
        json.dump(job2, f, ensure_ascii=False, indent=2)

    log("🎬 Starting Version 2: 848 × 480 (16:9 横屏)...")
    run(f"cd /content/colab-lipsync && python3 -u extracted_runner.py {job2_path} >> /content/generation_landscape.log 2>&1", "Execute Version 2 (848x480)")

    if os.path.exists(out2) and os.path.getsize(out2) > 1024 * 1024:
        log("✅ Version 2 complete! Video generated successfully.")
        with open("/content/STATUS_V2_DONE", "w") as f:
            f.write(f"DONE:{out2}\n")
    else:
        log("❌ Version 2 output file missing or invalid.")
        with open("/content/STATUS_V2_FAILED", "w") as f:
            f.write("FAILED\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"💥 Exception in master pipeline: {e}")
        with open("/content/STATUS_CRASHED", "w") as f:
            f.write(str(e))
        raise
