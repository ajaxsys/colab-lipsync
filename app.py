#!/usr/bin/env python3
"""
WebModel - InfiniteTalk (Wan2.1) Web 视频对口型工作室
支持多快捷词、按秒截取/整首切换、前端点击物理锁定、刷新自动回显请求参数与断线自愈监控
"""

import os
import sys
import json
import threading
import subprocess
import gradio as gr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER_PATH = os.path.join(CURRENT_DIR, "extracted_runner.py")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(CURRENT_DIR, "models"))
WORK_DIR = os.path.join(CURRENT_DIR, "workspace")
CODE_DIR = os.path.join(CURRENT_DIR, "InfiniteTalk")
JOB_FILE = os.path.join(WORK_DIR, "current_job.json")
PERF_FILE = "/content/perf.log"

os.makedirs(WORK_DIR, exist_ok=True)

JOB_LOCK = threading.Lock()
CURRENT_PROC = None


def is_runner_running():
    """检查后台是否有正在执行的 extracted_runner 进程"""
    try:
        out = subprocess.check_output(["ps", "-ef"], text=True)
        return any("extracted_runner.py" in line for line in out.splitlines())
    except Exception:
        return False


def get_history_videos():
    """获取工作目录下所有已生成的视频文件（按时间倒序）"""
    if not os.path.exists(WORK_DIR):
        return []
    files = [os.path.join(WORK_DIR, f) for f in os.listdir(WORK_DIR) if f.startswith("output_") and f.endswith(".mp4")]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files


def load_server_state():
    """页面刷新或加载时触发：从 current_job.json 回显所有表单参数与运行状态"""
    # 默认值
    img_val = None
    audio_val = None
    prompt_val = "女人正在唱歌"
    neg_val = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality"
    quality_val = "480P (快)"
    aspect_val = "原图比例"
    steps_val = 4
    seed_val = -1
    max_sec_val = "10"
    whole_song_val = False
    accel_val = True
    sage_val = True
    demucs_val = True
    video_val = None
    status_msg = "🟢 服务就绪，可随时提交新任务"

    # 1. 尝试从 current_job.json 恢复请求参数
    if os.path.exists(JOB_FILE):
        try:
            with open(JOB_FILE, "r", encoding="utf-8") as f:
                job = json.load(f)
            if job.get("image") and os.path.exists(job["image"]):
                img_val = job["image"]
            if job.get("audio") and os.path.exists(job["audio"]):
                audio_val = job["audio"]
            prompt_val = job.get("prompt", prompt_val)
            neg_val = job.get("negative", neg_val)
            steps_val = job.get("steps", steps_val)
            seed_val = job.get("seed", seed_val)
            accel_val = job.get("accel", accel_val)
            sage_val = job.get("sage", sage_val)
            demucs_val = ("vocals" not in job)
            
            # 时长与整首
            max_sec = job.get("max_seconds", 0)
            if max_sec and float(max_sec) > 0:
                max_sec_val = str(int(max_sec) if float(max_sec).is_integer() else max_sec)
                whole_song_val = False
            else:
                max_sec_val = "整首"
                whole_song_val = True

            # 输出文件
            if job.get("out") and os.path.exists(job["out"]):
                video_val = job["out"]
        except Exception as e:
            print("load_server_state error:", e)

    # 如果有历史视频且当前无视频，挂载最新历史视频
    if not video_val:
        hist = get_history_videos()
        if hist:
            video_val = hist[0]

    # 2. 检查进程是否还在运行
    running = is_runner_running()
    if running:
        last_log = "计算中…"
        if os.path.exists(PERF_FILE):
            lines = open(PERF_FILE, "r", encoding="utf-8", errors="ignore").readlines()
            if lines:
                last_log = lines[-1].strip()
        status_msg = f"⚡ 后台任务全速生成中 (已自动恢复会话接管)！最新心跳：{last_log}"
        btn_update = gr.Button(value="⏳ 视频生成中 (已锁定，请勿重复点击)...", interactive=False, variant="secondary")
    else:
        btn_update = gr.Button(value="🚀 开始生成", interactive=True, variant="primary")

    return (
        img_val, audio_val, prompt_val, neg_val, quality_val, aspect_val,
        steps_val, seed_val, max_sec_val, whole_song_val, accel_val, sage_val, demucs_val,
        video_val, btn_update, status_msg
    )


def build_job_config(image_path, audio_path, prompt, neg_prompt, quality, aspect_ratio, steps, seed, max_sec_str, is_whole_song, accel, sage, use_demucs):
    """根据前端选项生成符合 extracted_runner 规范的 job 字典"""
    is_hd = (quality == "720P (清晰，慢 2-3 倍)")
    if aspect_ratio == "9:16 竖屏":
        w, h = (480, 848) if not is_hd else (720, 1280)
    elif aspect_ratio == "16:9 横屏":
        w, h = (848, 480) if not is_hd else (1280, 720)
    elif aspect_ratio == "1:1":
        w, h = (640, 640) if not is_hd else (960, 960)
    elif aspect_ratio == "3:4":
        w, h = (480, 640) if not is_hd else (720, 960)
    elif aspect_ratio == "4:3":
        w, h = (640, 480) if not is_hd else (960, 720)
    else:  # 原图比例或默认
        w, h = (848, 480) if not is_hd else (1280, 720)

    # 截取时长解析
    max_seconds = 0
    if not is_whole_song and max_sec_str and str(max_sec_str).strip():
        s = str(max_sec_str).strip().lower()
        if s != "整首" and s != "all" and s != "0":
            try:
                max_seconds = float(s)
            except ValueError:
                pass

    out_file = os.path.join(WORK_DIR, f"output_{int(os.times().elapsed)}.mp4")

    job = {
        "work_dir": WORK_DIR,
        "code_dir": CODE_DIR,
        "image": image_path,
        "audio": audio_path,
        "prompt": prompt or "女人正在唱歌",
        "negative": neg_prompt or "",
        "out": out_file,
        "width": w,
        "height": h,
        "steps": int(steps) if steps else 4,
        "seed": int(seed) if seed and int(seed) >= 0 else -1,
        "max_seconds": max_seconds,
        "accel": bool(accel),
        "sage": bool(sage),
        "color_fix": 1.0,
        "models": {
            "base": os.path.join(MODELS_DIR, "base"),
            "dit": os.path.join(MODELS_DIR, "dit", "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"),
            "infinitetalk": os.path.join(MODELS_DIR, "infinitetalk", "infinitetalk_single.safetensors"),
            "lora": os.path.join(MODELS_DIR, "lora", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
            "wav2vec": os.path.join(MODELS_DIR, "wav2vec"),
            "t5_dir": os.path.join(MODELS_DIR, "t5"),
            "demucs": os.path.join(MODELS_DIR, "demucs", "htdemucs.pth")
        }
    }

    if not use_demucs:
        job["vocals"] = audio_path

    return job


def generate_video(image_path, audio_path, prompt, neg_prompt, quality, aspect_ratio, steps, seed, max_sec_str, is_whole_song, accel, sage, use_demucs, progress=gr.Progress()):
    global CURRENT_PROC

    if not image_path:
        raise gr.Error("请先上传人物照片！")
    if not audio_path:
        raise gr.Error("请先上传歌曲或音频文件！")

    acquired = JOB_LOCK.acquire(blocking=False)
    if not acquired:
        raise gr.Error("⚠️ 当前已有任务正在显卡上生成中！为防止显存溢出 (OOM)，请耐心等待上一任务完成。")

    # 1. 点击后瞬间锁定前端按钮
    yield None, gr.Button(value="⏳ 视频生成中 (已锁定，请勿重复点击)...", interactive=False, variant="secondary"), "🚀 正在初始化任务与模型…"

    try:
        job = build_job_config(image_path, audio_path, prompt, neg_prompt, quality, aspect_ratio, steps, seed, max_sec_str, is_whole_song, accel, sage, use_demucs)
        with open(JOB_FILE, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)

        cmd = [sys.executable, "-u", RUNNER_PATH, JOB_FILE]
        progress(0.01, desc="🚀 正在初始化显卡与计算引擎...")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )
        CURRENT_PROC = proc

        output_video = None
        error_msg = None
        status_line = "处理中…"

        for line in proc.stdout:
            line_clean = line.strip()
            if line_clean.startswith("@@LM"):
                try:
                    payload = json.loads(line_clean[4:].strip())
                    if "pct" in payload:
                        pct_val = payload["pct"]
                        pct = (float(pct_val) / 100.0) if pct_val is not None else 0.5
                        msg = payload.get("msg", "处理中…")
                        eta = payload.get("eta")
                        eta_str = f" (剩余约 {eta}s)" if eta else ""
                        status_line = f"{msg}{eta_str}"
                        progress(min(max(pct, 0.01), 0.99), desc=status_line)
                    elif "stage" in payload:
                        msg = payload.get("msg", payload["stage"])
                        status_line = f"【{payload['stage']}】{msg}"
                        progress(0.1, desc=status_line)
                    if "done" in payload:
                        output_video = payload["done"]
                    if "error" in payload:
                        error_msg = payload["error"]
                except Exception:
                    pass
            else:
                if line_clean:
                    print(line_clean)

        proc.wait()
        if proc.returncode != 0 or not output_video or not os.path.exists(output_video):
            raise gr.Error(f"生成中断或失败：{error_msg or '进程异常退出'}")

        progress(1.0, desc="🎉 视频生成完成！")
        yield output_video, gr.Button(value="🚀 开始生成", interactive=True, variant="primary"), "🎉 视频生成成功！"
    except Exception as e:
        yield None, gr.Button(value="🚀 开始生成", interactive=True, variant="primary"), f"❌ 任务出错：{str(e)}"
        if isinstance(e, gr.Error):
            raise e
        else:
            raise gr.Error(str(e))
    finally:
        CURRENT_PROC = None
        JOB_LOCK.release()


# ---------------- UI 界面定义 ----------------
custom_css = """
.container { max-width: 1000px; margin: auto; }
.quick-btn { margin-right: 6px; margin-bottom: 6px; }
.tip-text { font-size: 0.9em; color: #666; margin-top: 6px; }
.status-box { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; padding: 8px 12px; border-radius: 6px; font-size: 0.9em; margin-bottom: 10px; }
"""

with gr.Blocks(title="AI 对口型视频工作室 (InfiniteTalk)", css=custom_css) as demo:
    gr.Markdown("# 🎙️ InfiniteTalk 视频对口型工作室")
    status_banner = gr.Markdown("🟢 服务就绪", elem_classes=["status-box"])

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1. 描述 (可以写中文)")
            prompt_input = gr.Textbox(
                value="女人正在唱歌",
                label="",
                placeholder="请输入画面动态提示词，如：女人正在唱歌，面部生动，眼神自然流转",
                lines=2
            )
            
            with gr.Row():
                btn_woman_sing = gr.Button("女人正在唱歌", size="sm")
                btn_man_sing = gr.Button("男人正在唱歌", size="sm")
                btn_woman_talk = gr.Button("女人正在说话", size="sm")
                btn_man_talk = gr.Button("男人正在说话", size="sm")

            btn_woman_sing.click(fn=lambda: "女人正在唱歌，面部生动，眼神自然流转", outputs=[prompt_input])
            btn_man_sing.click(fn=lambda: "男人正在唱歌，面部生动，眼神自然流转", outputs=[prompt_input])
            btn_woman_talk.click(fn=lambda: "女人正在说话，表情自然，神态生动", outputs=[prompt_input])
            btn_man_talk.click(fn=lambda: "男人正在说话，表情自然，神态生动", outputs=[prompt_input])

            with gr.Accordion("▶ 不想出现的内容 (反向提示词，开加速时不起作用)", open=False):
                neg_input = gr.Textbox(
                    value="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality",
                    label="",
                    lines=2
                )

            gr.Markdown("### 2. 清晰度和画面")
            with gr.Row():
                quality_sel = gr.Radio(["480P (快)", "720P (清晰，慢 2-3 倍)"], value="480P (快)", label="清晰度")
                aspect_sel = gr.Radio(["原图比例", "9:16 竖屏", "16:9 横屏", "1:1", "3:4", "4:3"], value="原图比例", label="画面比例")

            with gr.Row():
                steps_input = gr.Number(value=4, label="步数", precision=0)
                seed_input = gr.Number(value=-1, label="种子 (-1 随机)", precision=0)
                max_sec_input = gr.Textbox(value="10", label="只生成前几秒", placeholder="填数字(如10)试跑", scale=2)
                whole_song_chk = gr.Checkbox(label="整首生成 (不截取)", value=False, scale=1)

            def on_whole_song_toggle(is_checked):
                if is_checked:
                    return gr.update(value="整首", interactive=False)
                else:
                    return gr.update(value="10", interactive=True)

            whole_song_chk.change(
                fn=on_whole_song_toggle,
                inputs=[whole_song_chk],
                outputs=[max_sec_input]
            )

            with gr.Row():
                accel_chk = gr.Checkbox(label="4 步加速 (lightx2v，推荐。关掉要 40 步，慢 10 倍以上)", value=True)
                sage_chk = gr.Checkbox(label="sage-attn 加速 (这台电脑用不了时会自动改用普通模式)", value=True)

            demucs_chk = gr.Checkbox(label="自动分离伴奏与人声 (使用 Demucs 提取清晰口型，推荐勾选)", value=True)

            gr.Markdown("### 3. 上传素材")
            with gr.Row():
                img_input = gr.Image(type="filepath", label="人物照片")
                audio_input = gr.Audio(type="filepath", label="歌曲或人声音频")

            gen_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
            gr.Markdown(
                """
                <div class="tip-text">
                💡 <b>贴心提示</b>：一首 3~4 分钟的完整歌曲需要分 10~20 段生成（耗时约 5~15 分钟）。<br>
                第一次建议保持「<b>只生成前几秒</b>」填 <b>10</b> 快速预览效果。点击生成后按钮会自动锁定，刷新页面也会自动回显！
                </div>
                """
            )

        with gr.Column(scale=4):
            gr.Markdown("### 4. 生成结果")
            video_output = gr.Video(label="输出视频", interactive=False)

            with gr.Accordion("📂 历史生成记录 (刷新页面后也可在此找回)", open=True):
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 刷新历史视频列表", size="sm")
                    history_dropdown = gr.Dropdown(label="选择历史生成的视频", choices=get_history_videos(), value=None)
                history_dropdown.change(fn=lambda v: v, inputs=[history_dropdown], outputs=[video_output])
                refresh_history_btn.click(fn=lambda: gr.update(choices=get_history_videos()), outputs=[history_dropdown])

    # 提交生成
    gen_btn.click(
        fn=generate_video,
        inputs=[
            img_input, audio_input, prompt_input, neg_input, quality_sel, aspect_sel,
            steps_input, seed_input, max_sec_input, whole_song_chk, accel_chk, sage_chk, demucs_chk
        ],
        outputs=[video_output, gen_btn, status_banner]
    )

    # 页面初次加载或刷新（F5 / Command+R）时自动回显上一次的参数与状态
    demo.load(
        fn=load_server_state,
        outputs=[
            img_input, audio_input, prompt_input, neg_input, quality_sel, aspect_sel,
            steps_input, seed_input, max_sec_input, whole_song_chk, accel_chk, sage_chk, demucs_chk,
            video_output, gen_btn, status_banner
        ]
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
