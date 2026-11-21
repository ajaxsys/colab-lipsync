#!/usr/bin/env python3
"""
WebModel - InfiniteTalk (Wan2.1) Web 视频对口型工作室
支持：
- 音频任意区间滑块选段与即点即听实时试听（截取试听无需消耗 GPU）
- 上传素材（图片、音频）全自动持久化与刷新回显（allowed_paths 支持）
- 点击生成物理锁定防重复提交与 GPU 排队保护
- 中文提示词快捷输入与反向提示词折叠面板
"""

import os
import sys
import json
import time
import shutil
import threading
import subprocess
import gradio as gr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER_PATH = os.path.join(CURRENT_DIR, "extracted_runner.py")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(CURRENT_DIR, "models"))
WORK_DIR = os.path.join(CURRENT_DIR, "workspace")
CODE_DIR = os.path.join(CURRENT_DIR, "InfiniteTalk")
JOB_FILE = os.path.join(WORK_DIR, "current_job.json")
META_FILE = os.path.join(WORK_DIR, "ui_metadata.json")
PERF_FILE = "/content/perf.log"

os.makedirs(WORK_DIR, exist_ok=True)

JOB_LOCK = threading.Lock()
CURRENT_PROC = None


def get_audio_duration(file_path):
    """获取音频文件的实际时长（秒）"""
    if not file_path or not os.path.exists(file_path):
        return 0.0
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            text=True
        ).strip()
        return float(out)
    except Exception:
        return 0.0


def format_time(seconds):
    """格式化秒数为 mm:ss 格式"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def save_ui_metadata(**kw):
    """持久化保存 UI 界面状态配置"""
    data = {}
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    data.update(kw)
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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
    """页面刷新或载入时触发：完整回显所有素材、滑块、参数与运行状态"""
    img_val = None
    audio_val = None
    prompt_val = "女人正在唱歌"
    neg_val = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality"
    quality_val = "480P (快)"
    aspect_val = "原图比例"
    steps_val = 4
    seed_val = -1
    start_sec_val = 0.0
    dur_sec_val = 10.0
    whole_song_val = False
    accel_val = True
    sage_val = True
    demucs_val = True
    video_val = None
    audio_info_msg = "尚未上传音频"
    status_msg = "🟢 服务就绪，可随时提交新任务"

    # 1. 尝试从 ui_metadata.json 与 current_job.json 恢复
    meta = {}
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    job = {}
    if os.path.exists(JOB_FILE):
        try:
            with open(JOB_FILE, "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception:
            pass

    # 恢复素材
    img_candidate = meta.get("image") or job.get("image")
    if img_candidate and os.path.exists(img_candidate):
        img_val = img_candidate

    audio_candidate = meta.get("audio") or job.get("audio")
    if audio_candidate and os.path.exists(audio_candidate):
        audio_val = audio_candidate
        dur = get_audio_duration(audio_val)
        audio_info_msg = f"🎵 音频已就绪，总时长：{format_time(dur)} ({dur:.1f} 秒)"

    prompt_val = meta.get("prompt") or job.get("prompt") or prompt_val
    neg_val = meta.get("negative") or job.get("negative") or neg_val
    quality_val = meta.get("quality", quality_val)
    aspect_val = meta.get("aspect", aspect_val)
    steps_val = meta.get("steps") or job.get("steps") or steps_val
    seed_val = meta.get("seed") or job.get("seed") or seed_val
    start_sec_val = meta.get("start_sec") or job.get("start_seconds") or start_sec_val
    dur_sec_val = meta.get("dur_sec") or job.get("max_seconds") or dur_sec_val
    whole_song_val = meta.get("whole_song", whole_song_val)
    accel_val = job.get("accel", accel_val)
    sage_val = job.get("sage", sage_val)
    demucs_val = ("vocals" not in job)

    # 恢复历史/最新视频
    hist = get_history_videos()
    if hist:
        video_val = hist[0]

    # 检查进程状态
    running = is_runner_running()
    if running:
        last_log = "计算中…"
        if os.path.exists(PERF_FILE):
            try:
                lines = open(PERF_FILE, "r", encoding="utf-8", errors="ignore").readlines()
                if lines:
                    last_log = lines[-1].strip()
            except Exception:
                pass
        status_msg = f"⚡ 后台任务全速生成中 (已自动恢复会话接管)！最新心跳：{last_log}"
        btn_update = gr.Button(value="⏳ 视频生成中 (已锁定，请勿重复点击)...", interactive=False, variant="secondary")
    else:
        btn_update = gr.Button(value="🚀 开始生成视频", interactive=True, variant="primary")

    slider_max = max(int(get_audio_duration(audio_val)), 60) if audio_val else 300

    return (
        img_val, audio_val, prompt_val, neg_val, quality_sel_val(quality_val), aspect_sel_val(aspect_val),
        steps_val, seed_val, gr.update(value=start_sec_val, maximum=slider_max),
        gr.update(value=dur_sec_val, interactive=not whole_song_val),
        whole_song_val, accel_val, sage_val, demucs_val,
        video_val, btn_update, status_msg, audio_info_msg
    )


def quality_sel_val(q):
    return "720P (清晰，慢 2-3 倍)" if "720" in str(q) else "480P (快)"


def aspect_sel_val(a):
    for choice in ["原图比例", "9:16 竖屏", "16:9 横屏", "1:1", "3:4", "4:3"]:
        if choice in str(a):
            return choice
    return "原图比例"


def handle_image_upload(img_path):
    """当用户上传图片时，自动持久化缓存到工作空间"""
    if not img_path or not os.path.exists(img_path):
        return None
    ext = os.path.splitext(img_path)[1] or ".png"
    target = os.path.join(WORK_DIR, f"cached_input_image{ext}")
    try:
        shutil.copy(img_path, target)
        save_ui_metadata(image=target)
        return target
    except Exception:
        return img_path


def handle_audio_upload(audio_path):
    """当用户上传音频时，自动分析时长、缓存并更新滑块范围"""
    if not audio_path or not os.path.exists(audio_path):
        return None, gr.update(), "尚未上传音频"
    ext = os.path.splitext(audio_path)[1] or ".wav"
    target = os.path.join(WORK_DIR, f"cached_input_audio{ext}")
    try:
        shutil.copy(audio_path, target)
    except Exception:
        target = audio_path

    dur = get_audio_duration(target)
    save_ui_metadata(audio=target, audio_dur=dur)
    max_slider = max(int(dur), 10)
    info_text = f"🎵 音频已就绪，总时长：{format_time(dur)} ({dur:.1f} 秒)"
    return target, gr.update(maximum=max_slider), info_text


def preview_audio_segment(audio_path, start_sec, dur_sec, is_whole_song):
    """毫秒级截取音频片段供网页试听（纯 ffmpeg 操作，无需显存与 GPU）"""
    target = audio_path
    if not target or not os.path.exists(target):
        # 尝试使用缓存
        candidates = [os.path.join(WORK_DIR, f) for f in os.listdir(WORK_DIR) if f.startswith("cached_input_audio")]
        if candidates and os.path.exists(candidates[0]):
            target = candidates[0]
        else:
            raise gr.Error("请先上传音频文件再进行试听！")

    if is_whole_song:
        return target

    start = max(float(start_sec or 0), 0.0)
    dur = max(float(dur_sec or 10), 1.0)
    preview_file = os.path.join(WORK_DIR, "preview_clip.wav")

    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", target,
        "-t", f"{dur:.3f}", "-c:a", "pcm_s16le", preview_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return preview_file
    except Exception as e:
        raise gr.Error(f"试听截取失败：{str(e)}")


def build_job_config(image_path, audio_path, prompt, neg_prompt, quality, aspect_ratio, steps, seed, start_sec, dur_sec, is_whole_song, accel, sage, use_demucs):
    """构建任务 JSON 配置"""
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
    else:
        w, h = (848, 480) if not is_hd else (1280, 720)

    # 截取区间
    start = 0.0 if is_whole_song else max(float(start_sec or 0), 0.0)
    max_seconds = 0.0 if is_whole_song else max(float(dur_sec or 10), 0.0)

    out_file = os.path.join(WORK_DIR, f"output_{int(time.time())}.mp4")

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
        "steps": int(steps),
        "seed": int(seed) if seed is not None and int(seed) >= 0 else -1,
        "start_seconds": start,
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


def generate_video(image_path, audio_path, prompt, neg_prompt, quality, aspect_ratio, steps, seed, start_sec, dur_sec, is_whole_song, accel, sage, use_demucs, progress=gr.Progress()):
    global CURRENT_PROC

    if not image_path or not os.path.exists(image_path):
        raise gr.Error("请先上传人物照片！")
    if not audio_path or not os.path.exists(audio_path):
        raise gr.Error("请先上传音频/歌曲！")

    acquired = JOB_LOCK.acquire(blocking=False)
    if not acquired:
        raise gr.Error("后台已有任务正在执行中，请等待上一个任务完成！")

    # 立即锁定按钮
    yield None, gr.Button(value="⏳ 视频生成中 (已锁定，请勿重复点击)...", interactive=False, variant="secondary"), "🟡 正在启动计算核心..."

    try:
        # 持久化元数据
        save_ui_metadata(
            image=image_path, audio=audio_path, prompt=prompt, negative=neg_prompt,
            quality=quality, aspect=aspect_ratio, steps=steps, seed=seed,
            start_sec=start_sec, dur_sec=dur_sec, whole_song=is_whole_song
        )

        job = build_job_config(image_path, audio_path, prompt, neg_prompt, quality, aspect_ratio, steps, seed, start_sec, dur_sec, is_whole_song, accel, sage, use_demucs)
        with open(JOB_FILE, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)

        cmd = [sys.executable, "-u", RUNNER_PATH, JOB_FILE]
        progress(0.01, desc="正在初始化模型与环境...")

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

        output_video, error_msg = None, None
        for line in proc.stdout:
            line_clean = line.strip()
            if line_clean.startswith("@@LM"):
                try:
                    payload = json.loads(line_clean[4:].strip())
                    if "pct" in payload:
                        pct = float(payload["pct"]) / 100.0
                        msg = payload.get("msg", "处理中…")
                        eta = payload.get("eta")
                        eta_str = f" (剩余约 {eta}s)" if eta else ""
                        progress(min(max(pct, 0.0), 1.0), desc=f"{msg}{eta_str}")
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
        yield output_video, gr.Button(value="🚀 开始生成视频", interactive=True, variant="primary"), "🎉 视频生成成功！"
    except Exception as e:
        yield None, gr.Button(value="🚀 开始生成视频", interactive=True, variant="primary"), f"❌ 任务出错：{str(e)}"
        if isinstance(e, gr.Error):
            raise e
        else:
            raise gr.Error(str(e))
    finally:
        CURRENT_PROC = None
        JOB_LOCK.release()


# ---------------- UI 界面定义 ----------------
custom_css = """
.container { max-width: 1050px; margin: auto; }
.card-box { background: #fafafa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
.status-box { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; padding: 10px 14px; border-radius: 8px; font-size: 0.95em; margin-bottom: 12px; font-weight: 500; }
.tip-text { font-size: 0.9em; color: #64748b; margin-top: 6px; line-height: 1.5; }
.audio-tag { color: #2563eb; font-weight: 600; font-size: 0.95em; margin-top: 4px; }
"""

with gr.Blocks(title="AI 对口型视频工作室 (InfiniteTalk)") as demo:
    gr.Markdown("# 🎙️ InfiniteTalk 视频对口型工作室")
    status_banner = gr.Markdown("🟢 服务就绪，已开启会话自愈与素材自动回显保护", elem_classes=["status-box"])

    with gr.Row():
        with gr.Column(scale=5):
            # 1. 描述
            with gr.Column(elem_classes=["card-box"]):
                gr.Markdown("### 1. 画面动态描述")
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

            # 2. 上传素材
            with gr.Column(elem_classes=["card-box"]):
                gr.Markdown("### 2. 上传素材 (刷新页面自动保留)")
                with gr.Row():
                    img_input = gr.Image(type="filepath", label="人物照片 (正面或微侧脸清晰照片)")
                    audio_input = gr.Audio(type="filepath", label="歌曲或人声音频 (支持 MP3/WAV/FLAC/M4A)")

                audio_info_display = gr.Markdown("尚未上传音频", elem_classes=["audio-tag"])

                # 素材上传即刻自动缓存
                img_input.change(fn=handle_image_upload, inputs=[img_input], outputs=[])

            # 3. 音频区间截取与试听
            with gr.Column(elem_classes=["card-box"]):
                gr.Markdown("### 3. 音频选段与精准试听")
                gr.Markdown("<div class='tip-text'>💡 拖动滑块定位到<b>有人声唱歌的片段</b>（避开漫长的前奏纯伴奏），点击【试听】即点即播，确认有歌词后再生成！</div>")

                with gr.Row():
                    start_sec_slider = gr.Slider(minimum=0.0, maximum=300.0, value=0.0, step=0.5, label="🎵 截取起始时间 (秒)")
                    dur_sec_slider = gr.Slider(minimum=3.0, maximum=60.0, value=10.0, step=1.0, label="⏱️ 截取持续时长 (秒)")
                
                whole_song_chk = gr.Checkbox(label="整首生成 (完整渲染整首歌曲，忽略上方截取滑块)", value=False)

                def on_whole_song_toggle(is_checked):
                    return gr.update(interactive=not is_checked), gr.update(interactive=not is_checked)

                whole_song_chk.change(
                    fn=on_whole_song_toggle,
                    inputs=[whole_song_chk],
                    outputs=[start_sec_slider, dur_sec_slider]
                )

                with gr.Row():
                    preview_btn = gr.Button("🎧 试听当前截取片段 (毫秒响应，不耗 GPU)", variant="secondary")
                preview_audio = gr.Audio(label="选定片段试听播放器", interactive=False)

                # 试听交互绑定
                preview_btn.click(
                    fn=preview_audio_segment,
                    inputs=[audio_input, start_sec_slider, dur_sec_slider, whole_song_chk],
                    outputs=[preview_audio]
                )

                # 音频改变时自动更新时长与滑块上限
                audio_input.change(
                    fn=handle_audio_upload,
                    inputs=[audio_input],
                    outputs=[audio_input, start_sec_slider, audio_info_display]
                )

            # 4. 生成参数
            with gr.Column(elem_classes=["card-box"]):
                gr.Markdown("### 4. 画质与进阶参数")
                with gr.Row():
                    quality_sel = gr.Radio(["480P (快)", "720P (清晰，慢 2-3 倍)"], value="480P (快)", label="清晰度")
                    aspect_sel = gr.Radio(["原图比例", "9:16 竖屏", "16:9 横屏", "1:1", "3:4", "4:3"], value="原图比例", label="画面比例")

                with gr.Row():
                    steps_input = gr.Number(value=4, label="扩散步数", precision=0)
                    seed_input = gr.Number(value=-1, label="随机种子 (-1 为随机)", precision=0)

                with gr.Row():
                    accel_chk = gr.Checkbox(label="4 步加速 (lightx2v 蒸馏加速，推荐开启)", value=True)
                    sage_chk = gr.Checkbox(label="sage-attn 注意力加速", value=True)
                    demucs_chk = gr.Checkbox(label="自动分离伴奏与人声 (Demucs)", value=True)

            gen_btn = gr.Button("🚀 开始生成视频", variant="primary", size="lg")

        with gr.Column(scale=4):
            gr.Markdown("### 5. 生成结果")
            video_output = gr.Video(label="输出视频", interactive=False)

            with gr.Accordion("📂 历史生成记录 (页面刷新后视频依然在此)", open=True):
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 刷新历史列表", size="sm")
                    history_dropdown = gr.Dropdown(label="选择历史生成的视频", choices=get_history_videos(), value=None)
                history_dropdown.change(fn=lambda v: v, inputs=[history_dropdown], outputs=[video_output])
                refresh_history_btn.click(fn=lambda: gr.update(choices=get_history_videos()), outputs=[history_dropdown])

    # 提交生成
    gen_btn.click(
        fn=generate_video,
        inputs=[
            img_input, audio_input, prompt_input, neg_input, quality_sel, aspect_sel,
            steps_input, seed_input, start_sec_slider, dur_sec_slider, whole_song_chk, accel_chk, sage_chk, demucs_chk
        ],
        outputs=[video_output, gen_btn, status_banner]
    )

    # 页面刷新（F5 / Command+R）时自动完整回显所有表单与历史状态
    demo.load(
        fn=load_server_state,
        outputs=[
            img_input, audio_input, prompt_input, neg_input, quality_sel, aspect_sel,
            steps_input, seed_input, start_sec_slider, dur_sec_slider, whole_song_chk, accel_chk, sage_chk, demucs_chk,
            video_output, gen_btn, status_banner, audio_info_display
        ]
    )


if __name__ == "__main__":
    demo.queue(concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        allowed_paths=[WORK_DIR, "/content", "/tmp"]
    )
