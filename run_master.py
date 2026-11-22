import os
import sys
import subprocess
import json
import time

def log(msg):
    t = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open("/content/master.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run(cmd, desc=""):
    log(f"=== [START] {desc} ===")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        log(f"=== [FAILED] {desc} (code {res.returncode}) ===")
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    log(f"=== [DONE] {desc} ===")

def main():
    log("🚀 Master pipeline started on Colab L4...")

    # 1. Clone repos if needed
    if not os.path.exists("/content/InfiniteTalk"):
        run("git clone https://github.com/MeiGen-AI/InfiniteTalk.git /content/InfiniteTalk", "Clone InfiniteTalk")

    if not os.path.exists("/content/colab-lipsync"):
        run("git clone https://github.com/ajaxsys/colab-lipsync.git /content/colab-lipsync", "Clone colab-lipsync")
    else:
        run("cd /content/colab-lipsync && git pull", "Git pull colab-lipsync")

    # 2. Install requirements
    run("pip install -q -r /content/colab-lipsync/requirements.txt", "Install requirements")

    # 3. Download models (will skip already downloaded models)
    run("cd /content/colab-lipsync && python3 download_models.py", "Download models")

    # Base models dict
    models_dict = {
        "base": "/content/models/base",
        "dit": "/content/models/dit/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        "infinitetalk": "/content/models/infinitetalk/infinitetalk_single.safetensors",
        "lora": "/content/models/lora/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "wav2vec": "/content/models/wav2vec",
        "t5_dir": "/content/models/t5",
        "demucs": "/content/models/demucs/htdemucs.pth"
    }

    # ========================================================
    # JOB 1: 720P 9:16 Portrait (720x1280) - 精准人物居中裁剪
    # 原图 1792x896。人物及话筒中心约在 x=996 处。
    # 截取框：x=744, y=0, w=504, h=896 (正好 9:16，人物脸部+话筒居中完美构图)
    # ========================================================
    os.makedirs("/content/workspace_portrait", exist_ok=True)
    out1 = "/content/output_720p_portrait_9_16.mp4"
    crop1 = {
        "x": 744.0 / 1792.0,
        "y": 0.0,
        "w": 504.0 / 1792.0,
        "h": 1.0
    }
    job1 = {
        "work_dir": "/content/workspace_portrait",
        "code_dir": "/content/InfiniteTalk",
        "image": "/content/girl.webp",
        "crop": crop1,
        "audio": "/content/song.mp3",
        "prompt": "女人正在唱歌，面部生动，眼神自然流转",
        "negative": "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        "out": out1,
        "res": "custom",
        "width": 720,
        "height": 1280,
        "steps": 4,
        "seed": 42,
        "start_seconds": 0.0,
        "max_seconds": 0.0,
        "accel": True,
        "sage": True,
        "color_fix": 1.0,
        "models": models_dict
    }

    job1_path = "/content/workspace_portrait/job_portrait.json"
    with open(job1_path, "w", encoding="utf-8") as f:
        json.dump(job1, f, ensure_ascii=False, indent=2)

    log("🎬 Starting Job 1: 720P 9:16 Portrait (Centered)...")
    run(f"cd /content/colab-lipsync && python3 -u extracted_runner.py {job1_path} >> /content/generation_portrait.log 2>&1", "Execute Job 1 (720P 9:16)")

    if os.path.exists(out1) and os.path.getsize(out1) > 1024 * 1024:
        log("✅ Job 1 complete! Video generated successfully.")
        with open("/content/STATUS_JOB1_DONE", "w") as f:
            f.write(f"DONE:{out1}\n")
    else:
        log("❌ Job 1 output file missing or invalid.")
        with open("/content/STATUS_JOB1_FAILED", "w") as f:
            f.write("FAILED\n")
        return

    # ========================================================
    # JOB 2: 720P 16:9 Landscape (1280x720)
    # ========================================================
    os.makedirs("/content/workspace_landscape", exist_ok=True)
    out2 = "/content/output_720p_landscape_16_9.mp4"
    job2 = {
        "work_dir": "/content/workspace_landscape",
        "code_dir": "/content/InfiniteTalk",
        "image": "/content/girl.webp",
        "audio": "/content/song.mp3",
        "prompt": "女人正在唱歌，面部生动，眼神自然流转",
        "negative": "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        "out": out2,
        "res": "custom",
        "width": 1280,
        "height": 720,
        "steps": 4,
        "seed": 42,
        "start_seconds": 0.0,
        "max_seconds": 0.0,
        "accel": True,
        "sage": True,
        "color_fix": 1.0,
        "models": models_dict
    }

    job2_path = "/content/workspace_landscape/job_landscape.json"
    with open(job2_path, "w", encoding="utf-8") as f:
        json.dump(job2, f, ensure_ascii=False, indent=2)

    log("🎬 Starting Job 2: 720P 16:9 Landscape...")
    run(f"cd /content/colab-lipsync && python3 -u extracted_runner.py {job2_path} >> /content/generation_landscape.log 2>&1", "Execute Job 2 (720P 16:9)")

    if os.path.exists(out2) and os.path.getsize(out2) > 1024 * 1024:
        log("✅ Job 2 complete! Video generated successfully.")
        with open("/content/STATUS_ALL_DONE", "w") as f:
            f.write(f"DONE:{out2}\n")
    else:
        log("❌ Job 2 output file missing or invalid.")
        with open("/content/STATUS_JOB2_FAILED", "w") as f:
            f.write("FAILED\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"💥 Exception in master pipeline: {e}")
        with open("/content/STATUS_CRASHED", "w") as f:
            f.write(str(e))
        raise
