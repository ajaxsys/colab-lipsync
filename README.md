# 🎙️ Colab InfiniteTalk Lipsync (Wan2.1)

基于 **Wan2.1 (14B) + InfiniteTalk + LightX2V 蒸馏加速** 的视频对口型工作室 WebUI。针对 Google Colab（A100 / L4）深度定制，提供免安装一键运行 Notebook、防刷新状态回显与 Google Drive 模型极速备份方案。

---

## ✨ 核心特性

- **一键开箱即用**：提供标准 Jupyter Notebook，一行代码自动配置依赖、模型与穿透网络。
- **页面刷新自动回显**：前端与后端状态彻底解耦，无论何时刷新浏览器，已上传的图片、音频、参数设置以及生成的视频均自动恢复回显。
- **物理防重复锁**：点击生成后按钮自动锁定并禁用，后台任务严格单并发排队，杜绝误触多点导致 GPU 显存溢出（OOM）。
- **智能截取控制**：支持自由指定“只生成前 X 秒”进行快速预览，或一键勾选“整首生成”。
- **云盘离线兜底备份（可选）**：支持将模型权重归档备份至个人 Google Drive 作为弱网/断网兜底；平时推荐直接通过官方高速 CDN 2~3 分钟直载进 NVMe 高速本地盘。
- **静音/前奏保护**：内置人声分离门限保护，针对前奏纯伴奏与静音片段，防止底噪被放大导致误张嘴。

---

## 💡 硬件配置与最佳黄金“甜蜜点” (Sweet Spot)

经过大量实测与算力开销对比，我们总结出运行 InfiniteTalk 的最佳硬件配置策略：

| GPU 规格 | 显存 / 内存 | 单步推理速度 | 480P 10s 耗时 | 推荐指数 | 场景与性价比分析 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **NVIDIA A100 (40GB Standard)** | **40GB HBM2e / 83GB RAM** | **⚡ 13.3 秒 / 步** | **~40 秒** | 🌟🌟🌟🌟🌟<br/>**(最佳黄金甜蜜点)** | **首选主力卡**：40 层模型全常驻显存，零层交换延迟；点数仅需 ~12 点/hr（比 80GB 省近一半）；卡槽充裕易申请。 |
| **NVIDIA L4** | 24GB GDDR6 / 52GB RAM | **~45 秒 / 步** | **~2-3 分钟** | 🌟🌟🌟🌟<br/>(高性价比备选) | **经济长效**：点数消耗更低，适合长时间挂机或日常调试；Layer Swapper 自动调优。 |
| **NVIDIA A100 (80GB High-RAM)** | 80GB / 85GB RAM | 13.3 秒 / 步 | ~40 秒 | 🌟🌟<br/>(不推荐争抢) | 资源严重溢出，每小时消耗高达 18~23 点，且常年排队报 `Service Unavailable`，**完全无必要申请**。 |
| **NVIDIA T4** | 16GB / 12.7GB RAM | OOM | 崩溃 | ❌ (不可用) | 显存与架构无法承载 14B 模型，请勿使用。 |

> [!TIP]
> ### 🎯 为什么 A100-40GB 是终极“黄金甜蜜点”？
> 1. **显存刚好完美吃满**：在 480P 原生高清模式下，Wan2.1 14B + InfiniteTalk + VAE 峰值显存仅占用 **26.9 GB**。40GB 显存刚好让所有 40/40 层全部驻留显卡，达到 312 TFLOPS 的满血算力，**速度和 80GB 版本完全没有任何区别**！
> 2. **巨省算力点数**：A100-40GB（Standard 档）每小时消耗约 **12.5 计算单元**，而 80GB High-RAM 档每小时高达 **18~23 计算单元**，点数直接节省 **40% ~ 50%**！
> 3. **秒级秒批，无需排队**：全网都在争抢 80GB High-RAM，导致经常遇到 `Service Unavailable`；而 40GB Standard 算力池卡槽充裕，秒申秒就绪。
> 4. **自带 83GB 巨量内存**：Colab 为 A100 Standard 默认标配了 **83GB 系统内存**，本身就已处于 High-RAM 规格，根本不需要额外勾选 `--high-mem`。

---

## 🚀 快速开始 (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajaxsys/colab-lipsync/blob/main/Colab_InfiniteTalk_Web.ipynb)

直接点击上方徽章或打开项目内的 [`Colab_InfiniteTalk_Web.ipynb`](Colab_InfiniteTalk_Web.ipynb)，在 Colab 中一键运行：

### 1. 检查显卡
确保 Colab 运行模式选择为 **GPU** (推荐 A100-40GB 或 L4)：
```bash
!nvidia-smi
```

### 2. 克隆项目与安装依赖
```bash
!git clone https://github.com/ajaxsys/colab-lipsync.git /content/colab-lipsync
%cd /content/colab-lipsync
!pip install -q -r requirements.txt
!git clone https://github.com/MeiGen-AI/InfiniteTalk.git /content/InfiniteTalk || true
```

### 3. 准备模型权重（推荐官方 CDN 直载）
运行下载脚本，通过千兆高速 CDN 仅需约 2~3 分钟即可将全部权重（Wan2.1 fp8, InfiniteTalk, LoRA, wav2vec, Demucs）直接写入 NVMe 高速本地盘：
```bash
python download_models.py
```
*(注：Google Drive 备份仅作为可选的离线兜底选项，日常使用直接 CDN 下载更快)*

### 4. 启动 WebUI 与穿透隧道
```bash
python app.py
```

---

## 📂 项目文件结构

```text
├── Colab_InfiniteTalk_Web.ipynb   # Colab 一键运行 Notebook
├── app.py                         # Gradio WebUI 前端交互服务 (含回显与锁定机制)
├── extracted_runner.py            # 核心推理引擎 (Wan2.1 + InfiniteTalk + 内存动态交换)
├── download_models.py             # 29GB 模型多线程下载与路径映射脚本
├── requirements.txt               # 项目依赖清单
└── README.md                      # 项目说明文档
```

---

## 📄 开源与致谢
- [Wan2.1 (Wan-AI)](https://github.com/Wan-Video/Wan2.1)
- [InfiniteTalk (MeiGen-AI)](https://github.com/MeiGen-AI/InfiniteTalk)
- [LightX2V](https://github.com/Kijai/WanVideo_comfy)
