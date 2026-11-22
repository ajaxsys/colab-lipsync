#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeFormer 面部超分辨率与五官画质增强模块 (Face Super-Resolution & Restoration)
用于消除 Wan2.1 / 数字人生成中的面部水彩感、细节模糊，提升眼神与唇部清晰度。
支持独立命令行使用，亦支持作为 extracted_runner.py 后的自动化后处理流水线。
"""

import os
import sys
import argparse
import subprocess
import shutil
import tempfile


def run_cmd(cmd, desc=""):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{desc} 失败 (code {res.returncode}): {res.stderr}")
    return res.stdout


def ensure_codeformer_weights(models_dir="/content/models/codeformer"):
    """确保 CodeFormer 及依赖检测权重存在，若无则自动从 Hugging Face 下载"""
    os.makedirs(models_dir, exist_ok=True)
    weight_path = os.path.join(models_dir, "codeformer.pth")
    if not os.path.exists(weight_path):
        print("📥 正在下载 CodeFormer 预训练权重 (~376MB)...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="sczhou/CodeFormer", filename="codeformer.pth", local_dir=models_dir)
        except Exception as e:
            url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            import urllib.request
            urllib.request.urlretrieve(url, weight_path)
    return weight_path


def enhance_video_faces(input_video, output_video, fidelity=0.7, scale=1, models_dir=None):
    """
    对视频执行面部超分辨率与纹理修复
    :param input_video: 输入 mp4 文件路径
    :param output_video: 增强后的输出 mp4 文件路径
    :param fidelity: 保真度权重 [0.0 ~ 1.0]，默认 0.7 (数值越大越接近原人脸特征，数值越小越清晰/磨皮)
    :param scale: 放大倍数 (默认 1，保持原分辨率)
    :param models_dir: 权重存储目录
    """
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"输入视频不存在: {input_video}")

    if models_dir is None:
        models_dir = os.environ.get("MODELS_DIR", "/content/models")
        models_dir = os.path.join(models_dir, "codeformer")

    codeformer_pth = ensure_codeformer_weights(models_dir)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("⚠️ 未检测到 opencv-python 依赖，正在执行动态 pip 安装...")
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "opencv-python"], "安装 opencv-python")
        import cv2
        import numpy as np

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✨ 正在初始化 CodeFormer (设备: {device}, 保真度 fidelity: {fidelity})...")

    # 尝试加载 facexlib 与 CodeFormer
    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from torchvision.transforms.functional import normalize
    except ImportError:
        print("⚠️ 未检测到 facexlib 依赖，正在执行动态 pip 安装...")
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "facexlib", "basicsr"], "安装 facexlib")
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from torchvision.transforms.functional import normalize

    # 初始化人脸对齐与裁剪器
    face_helper = FaceRestoreHelper(
        scale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device
    )

    # 动态构建/载入 CodeFormer 架构
    try:
        from basicsr.utils import registry
        # 如果已经存在或者可以直接加载
        from codeformer.archs.codeformer_arch import CodeFormer
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(device)
    except Exception:
        # 降级：从 TorchHub 或动态定义
        import torch.nn as nn
        # 若环境未克隆官方 CodeFormer 仓库，通过官方 torch 加载
        checkpoint = torch.load(codeformer_pth, map_location=device)
        # 尝试标准载入
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(device)
            net.load_state_dict(checkpoint["params_ema"])
        except Exception:
            print("⚠️ 尝试使用 CodeFormer Standalone 模式载入...")
            raise RuntimeError("CodeFormer 模型定义加载失败，请确保 codeformer 或 basicsr 可用")

    net.eval()

    # 打开输入视频
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_dir = tempfile.mkdtemp(prefix="cf_enhance_")
    temp_raw_video = os.path.join(temp_dir, "enhanced_temp.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(temp_raw_video, fourcc, fps, (w * scale, h * scale))

    frame_idx = 0
    print(f"🎬 开始逐帧增强视频 (共 {total_frames} 帧, {w}x{h} @ {fps:.1f}fps)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 25 == 0 or frame_idx == total_frames:
            print(f"  ⚡ 已处理 {frame_idx}/{total_frames} 帧 ({(frame_idx/max(total_frames,1)*100):.1f}%)...")

        face_helper.clean_all()
        face_helper.read_image(frame)
        # 检测人脸
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        face_helper.align_warp_face()

        # 逐张人脸高清重建
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # 转换为 tensor
            cropped_face_t = torch.from_numpy(cropped_face).permute(2, 0, 1).float()[None] / 255.0
            cropped_face_t = normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).to(device)

            with torch.no_grad():
                output = net(cropped_face_t, w=fidelity, adain=True)[0]
                restored_face = output.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
                restored_face = ((restored_face * 0.5 + 0.5) * 255.0).astype(np.uint8)

            face_helper.add_restored_face(restored_face)

        # 贴回原始大图
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()
        out_writer.write(restored_img)

    cap.release()
    out_writer.release()
    print("✅ 画面逐帧修复完成，正在复用原音频音轨进行精确封包...")

    # 使用 ffmpeg 将原音频完整复制合并
    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", temp_raw_video,
        "-i", input_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_video
    ]
    run_cmd(ffmpeg_cmd, "合并原音轨")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"🎉 CodeFormer 面部增强视频已生成: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="CodeFormer Face Enhancement Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="增强后输出视频路径")
    parser.add_argument("--fidelity", "-w", type=float, default=0.7, help="人脸保真度 (0.0~1.0，默认 0.7)")
    parser.add_argument("--scale", "-s", type=int, default=1, help="分辨率放大倍数 (默认 1 不缩放)")
    parser.add_argument("--models_dir", type=str, default=None, help="模型缓存目录")
    args = parser.parse_args()

    enhance_video_faces(args.input, args.output, fidelity=args.fidelity, scale=args.scale, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
